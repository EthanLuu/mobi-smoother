import numpy as np
import gym
import random
from math import floor, ceil
from gym import spaces

VideoRatioIdxMap = {0: "360p", 1: "480p", 2: "720p", 3: "1080p"}

UpsampleRatioIdxMap = {0: "x1", 1: "x2", 2: "x4"}


class VideoItem:

    def __init__(self, base_size, total_time):
        # 0.7 : 1 : 2.2 : 3.2
        self.size_list = [
            base_size * 0.7, base_size, base_size * 2.2, base_size * 3.2
        ]
        self.buffer_list = [0] * len(self.size_list)
        self.total_time = total_time
        self.selected_action = -1
        self.upsample_ratio = 0

    def upsample(self, upsample_ratio):
        upsmaple_speed_map = {
            0: [0, 50, 40],
            1: [0, 35, 30],
            2: [0, 15, 10],
            3: [0, 8, 5]
        }
        self.upsample_ratio = upsample_ratio
        if upsample_ratio == 0:
            return 0
        upsample_speed = upsmaple_speed_map[self.selected_action]
        return self.total_time * 30 / upsample_speed[upsample_ratio]

    def reset_buffer(self):
        self.buffer_list = [0] * len(self.size_list)
        self.selected_action = -1

    def buffer(self, bandwidth, action):
        # 只选择一种分辨率缓冲到底
        if self.selected_action >= 0:
            action = self.selected_action
        else:
            self.selected_action = action
        size = self.size_list[action]
        # 返回缓冲视频花费的时间
        buffered_time = ceil(size / bandwidth)
        return buffered_time

    def __str__(self):
        return f"Size: {round(self.size_list[self.selected_action], 2)}, Total time: {self.total_time}, Video ratio: {VideoRatioIdxMap[self.selected_action]}, Upsample ratio: {UpsampleRatioIdxMap[self.upsample_ratio]}"


class LoadVideoEnv(gym.Env):
    observation_space: spaces.Box
    action_space: spaces.Discrete
    video_list: list
    min_bandwidth_demand: float
    max_bandwidth_demand: float

    def __init__(self, bandwidth_list=[], video_list=[]):
        super(LoadVideoEnv).__init__()
        # 行为空间：[分辨率选择(0-3), 超分倍数]
        # self.action_space = spaces.MultiDiscrete([4, 3])
        self.action_space = spaces.Discrete(12)

        # 不同分辨率的视频有不同的奖励系数
        # size: 0.7 : 1 : 2.2 : 3.2
        self.reward_ratio = [1, 2.3, 6, 10]
        self.upsample_reward_ratio = [1, 2, 2.6]

        self.max_bandwidth_demand = 0
        self.min_bandwidth_demand = float("inf")
        self.video_list = video_list if len(
            video_list) > 0 else self.generate_random_video_list(200)
        self.bandwidth_list = bandwidth_list if len(
            bandwidth_list) > 0 else self.generate_random_brandwidth_trace(30)

        # 观察空间：[上次的网络带宽, 下个视频480p需要的带宽, 可用于缓冲的时间]
        low_bound = [min(self.bandwidth_list), self.min_bandwidth_demand, 0]
        high_bound = [
            max(self.bandwidth_list), self.max_bandwidth_demand, 1000
        ]
        self.observation_space = spaces.Box(low=np.array(low_bound,
                                                         dtype=np.float32),
                                            high=np.array(high_bound,
                                                          dtype=np.float32),
                                            shape=(3, ),
                                            dtype=np.float32)

        self.video_list_len = len(self.video_list)

        self.cur_bandwidth_index = 0
        self.buffer_video_index = 0
        self.buffered_total_time = 0
        self.cur_time = 0
        self.reward = 0
        self.done = False
        self.pre_action = -1
        self.cur_bandwidth = self.bandwidth_list[self.cur_bandwidth_index]

    def reset(self, seed=0):
        if seed > 0: self.seed(seed)
        for video_item in self.video_list:
            video_item.reset_buffer()
        # random.shuffle(self.video_list)
        self.cur_bandwidth_index = 0
        self.buffer_video_index = 0
        self.buffered_total_time = 0
        self.cur_time = 0
        self.reward = 0
        self.done = False
        self.pre_action = -1
        self.cur_bandwidth = self.bandwidth_list[self.cur_bandwidth_index]
        return self._get_observation()

    def encode_action(self, video_ratio, upsample_ratio):
        return video_ratio * 3 + upsample_ratio

    def decode_action(self, action: int):
        # 0 -> 360p + x1
        # 1 -> 360p + x2
        # 2 -> 360p + x4
        return floor(action / 3), action % 3

    def step(self, action: int):
        # 每个 step 尝试缓冲一个视频
        video_ratio, upsample_ratio = self.decode_action(action)

        # 先缓冲
        buffer_video = self.video_list[self.buffer_video_index]
        buffer_consumption = buffer_video.buffer(self.cur_bandwidth,
                                                 video_ratio)
        self.cur_time += buffer_consumption
        self.buffered_total_time += buffer_video.total_time

        # 后超分
        upsample_time = buffer_video.upsample(upsample_ratio)
        self.cur_time += upsample_time

        reward = self._get_reward()
        done = self._get_done()
        obs = self._get_observation()
        info = self._get_info()
        truncated = False

        self.cur_bandwidth_index = (self.cur_bandwidth_index + 1) % len(
            self.bandwidth_list)
        self.cur_bandwidth = self.bandwidth_list[self.cur_bandwidth_index]
        self.buffer_video_index += 1

        return obs, reward, done, truncated, info

    def _get_reward(self):
        # 计算加载当前视频获取的 reward
        buffered_video = self.video_list[self.buffer_video_index]
        ratio = self.reward_ratio[buffered_video.selected_action]
        if buffered_video.upsample_ratio > 0:
            ratio *= self.upsample_reward_ratio[buffered_video.upsample_ratio]
        elif self.pre_action > buffered_video.selected_action:
            ratio *= 0.7
        self.pre_action = buffered_video.selected_action
        self.reward = ratio * buffered_video.total_time / 40

        # 网络带宽低/超分耗时，观看发生卡顿
        if self.buffered_total_time < self.cur_time:
            diff = self.cur_time - self.buffered_total_time
            self.buffered_total_time = self.cur_time
            self.reward = -diff / 10

        return round(self.reward, 2)

    def _get_done(self):
        self.done = self.buffer_video_index == self.video_list_len - 1
        return self.done

    def _get_observation(self):
        # 观察空间包括 [上次的网络带宽, 下个视频480p需要的带宽, 可用于缓冲的时间]
        obs = [self.cur_bandwidth]
        buffer_video = self.video_list[-1]
        if not self.done:
            buffer_video = self.video_list[self.buffer_video_index]
        obs.append(buffer_video.size_list[1] / buffer_video.total_time)
        obs.append(self.buffered_total_time - self.cur_time)
        return np.array(obs, dtype=np.float32)

    def close(self):
        return

    def get_action_cnt_map(self):
        # 统计每种分辨率的选择
        buffers = {"360p": 0, "480p": 0, "720p": 0, "1080p": 0}
        upsamples = {"x1": 0, "x2": 0, "x4": 0}
        for video in self.video_list:
            if video.selected_action < 0:
                break
            buffers[VideoRatioIdxMap[video.selected_action]] += 1
            upsamples[UpsampleRatioIdxMap[video.upsample_ratio]] += 1
        return buffers, upsamples

    def _get_info(self):
        buffer_video = self.video_list[self.buffer_video_index]
        info = {
            "Bandwidth": self.cur_bandwidth,
            "Buffered time": self.buffered_total_time,
            "Current time": self.cur_time,
            f"Video {self.buffer_video_index}": str(buffer_video)
        }
        return info

    def get_random_reward(self):
        self.reset()
        rewards = []
        while not self.done:
            action = self.action_space.sample()
            obs, reward, done, truncated, info = self.step(action)
            rewards.append(reward)
        self.reset()
        return sum(rewards)

    def get_greedy_reward(self):
        self.reset()
        rewards = []
        while not self.done:
            buffer_video = self.video_list[self.buffer_video_index]
            cur_bandwidth = self.cur_bandwidth
            total_time = buffer_video.total_time
            size_list = buffer_video.size_list
            video_ratio = 0
            for size in size_list[::-1]:
                if size / total_time <= cur_bandwidth:
                    video_ratio = size_list.index(size)
                    break
            obs, reward, done, truncated, info = self.step(
                self.encode_action(video_ratio, 0))
            rewards.append(reward)
        self.reset()
        return sum(rewards)

    def generate_random_video_list(self, l: int) -> list[VideoItem]:
        # 480p 1s 100KB
        video_list = []
        for _ in range(l):
            # 10-60s 视频
            video_time = random.randint(10, 60)
            video_size = video_time * (95 + 10 * random.random())
            bandwidth_demand = video_size / video_time
            self.max_bandwidth_demand = max(self.max_bandwidth_demand,
                                            bandwidth_demand)
            self.min_bandwidth_demand = max(self.min_bandwidth_demand,
                                            bandwidth_demand)
            video_item = VideoItem(video_size, video_time)
            video_list.append(video_item)
        return video_list

    def generate_random_brandwidth_trace(self, l: int) -> list[float]:
        # 根据缓冲 480p 视频所需要的带宽来确定基础带宽
        trace = []
        video = self.video_list[0]
        video_size = video.size_list[1]
        video_time = video.total_time
        base = video_size / video_time

        cur = base
        for _ in range(l):
            # 每次缓冲网络带宽发生 +/- 20% 的变化
            cur = cur * (70 + random.random() * 60) / 100
            if cur <= 0.2:
                cur = base
            elif cur >= 4 * base:
                cur = base
            trace.append(round(cur, 2))
        return trace


if __name__ == '__main__':
    env = LoadVideoEnv()
    env.reset()
    rewards = []
    for _ in range(10):
        action = env.action_space.sample()
        print(action)
        obs, reward, done, truncated, info = env.step(action)
        print(f"R: {reward}, " + str(list(obs)))
        rewards.append(reward)
        if done:
            print(
                f'Done! PPO Rewards: {sum(rewards):.2f}, Random Rewards: {env.get_random_reward():.2f}, Greedy Rewards: {env.get_greedy_reward():.2f}'
            )
            env.reset()
            rewards.clear()