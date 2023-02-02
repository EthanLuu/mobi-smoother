import numpy as np
import gym
import random
from math import floor, ceil
from gym import spaces


class VideoItem:

    def __init__(self, size_list, total_time):
        self.size_list = size_list
        self.buffer_list = [0] * len(size_list)
        self.total_time = total_time
        self.selected_action = -1

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
        return f"Size: {self.size_list[self.selected_action]}, Total time: {self.total_time}, Ratio: {self.selected_action}"


class LoadVideoEnv(gym.Env):
    observation_space: spaces.Box
    action_space: spaces.Discrete
    video_list: list

    def __init__(self, bandwidth_list=[], video_list=[]):
        super(LoadVideoEnv).__init__()
        # 行为空间：4 种分辨率，360/480/720/1080
        self.action_space = spaces.Discrete(4)

        # 不同分辨率的视频有不同的奖励系数
        self.reward_ratio = [1, 2.5, 6, 14]

        self.video_list = video_list if len(
            video_list) > 0 else self.generate_random_video_list(100)
        self.bandwidth_list = bandwidth_list if len(
            bandwidth_list) > 0 else self.generate_random_brandwidth_trace(30)

        # 观察空间：当前的带宽，可以用来缓冲的时间，待缓冲视频的视频720p对应的数据量，待缓冲视频时长
        self.observation_space = spaces.Box(low=np.array([0.2, 0, 36, 6],
                                                         dtype=np.float32),
                                            high=np.array([20, 100, 108, 20],
                                                          dtype=np.float32),
                                            shape=(4, ),
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

    def step(self, action: int):
        # 每个 step 尝试缓冲一个视频
        # 计算缓冲完待缓冲视频需要的时间
        buffer_video = self.video_list[self.buffer_video_index]
        buffer_consumption = buffer_video.buffer(self.cur_bandwidth, action)
        self.cur_time += buffer_consumption
        self.buffered_total_time += buffer_video.total_time

        reward = self._get_reward()
        done = self._get_done()
        obs = self._get_observation()
        info = self._get_info()
        truncated = False

        return obs, reward, done, truncated, info

    def _get_reward(self):
        # 缓冲完视频后进入下一个视频
        buffered_video = self.video_list[self.buffer_video_index]
        ratio = self.reward_ratio[buffered_video.selected_action]
        if self.pre_action < buffered_video.selected_action:
            ratio *= 0.9
        elif self.pre_action > buffered_video.selected_action:
            ratio *= 0.6
        self.pre_action = buffered_video.selected_action
        self.reward = ratio * buffered_video.total_time / 30
        self.buffer_video_index += 1

        # 缓冲速度偏慢，观看发生卡顿
        if self.buffered_total_time < self.cur_time:
            diff = self.cur_time - self.buffered_total_time
            self.buffered_total_time = self.cur_time
            self.reward = -diff

        return round(self.reward, 2)

    def _get_done(self):
        self.done = self.buffer_video_index == self.video_list_len
        return self.done

    def _get_observation(self):
        # 观察空间包括 [上次的网络带宽, 可以用来缓冲下一个视频的时间, 待缓冲视频的720p对应的数据大小, 待缓冲视频的时间长度]
        obs = [self.cur_bandwidth, self.buffered_total_time - self.cur_time]
        buffer_video = self.video_list[-1]
        self.cur_bandwidth_index = (self.cur_bandwidth_index + 1) % len(
            self.bandwidth_list)
        if not self.done:
            buffer_video = self.video_list[self.buffer_video_index]
        obs.extend([buffer_video.size_list[-2], buffer_video.total_time])
        return np.array(obs, dtype=np.float32)

    def close(self):
        return

    def get_action_cnt_map(self):
        # 统计每种分辨率的选择
        map = {0: 0, 1: 0, 2: 0, 3: 0}
        for video in self.video_list:
            if video.selected_action < 0:
                break
            map[video.selected_action] += 1
        return map

    def _get_info(self):
        buffer_video = self.video_list[self.buffer_video_index - 1]
        info = {
            "Bandwidth": self.cur_bandwidth,
            "Buffered time": self.buffered_total_time,
            "Current time": self.cur_time,
            f"Buffer {self.buffer_video_index}": str(buffer_video)
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
            action = 0
            for size in size_list[::-1]:
                if size / total_time <= cur_bandwidth:
                    action = size_list.index(size)
                    break
            obs, reward, done, truncated, info = self.step(action)
            rewards.append(reward)
        self.reset()
        return sum(rewards)

    def generate_random_video_list(self,
                                   video_list_len: int) -> list[VideoItem]:
        video_list = []
        for _ in range(video_list_len):
            random_size = floor(random.random() * 6 + 12) * 6  # Size: 72 ~ 216
            random_time = random_size // 12 + random.randint(0,
                                                             5)  # Time: 6 ~ 15
            video_item = VideoItem([
                random_size / 6, random_size / 3, random_size / 2, random_size
            ], random_time)
            video_list.append(video_item)
        return video_list

    def generate_random_brandwidth_trace(self, l: int) -> list[VideoItem]:
        # 根据缓冲 720p 视频所需要的带宽来确定基础带宽
        trace = []
        video = self.video_list[0]
        video_size = video.size_list[2]
        video_time = video.total_time
        base = video_size / video_time

        cur = base
        for _ in range(l):
            if cur <= 0.2:
                cur = base
            elif cur >= 3 * base:
                cur = base
            # 每次缓冲网络带宽发生 [-3, 3] 的变化
            cur += random.random() * 6 - 3
            trace.append(round(cur, 2))
        return trace


if __name__ == '__main__':
    env = LoadVideoEnv()
    env.reset()
    rewards = []
    for _ in range(250):
        obs, reward, done, truncated, info = env.step(
            env.action_space.sample())
        # print(f"R: {reward}, " + str(list(obs)))
        rewards.append(reward)
        if done:
            print(
                f'Done! PPO Rewards: {sum(rewards):.2f}, Random Rewards: {env.get_random_reward():.2f}, Greedy Rewards: {env.get_greedy_reward():.2f}'
            )
            env.reset()
            rewards.clear()