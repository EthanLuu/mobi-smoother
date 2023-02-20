# from models import PPO, Agent
from common.utils import merge_class_attrs, plot_rewards, save_cfgs, save_results
from config.config import DefaultConfig
from envs.register import register_env
from utils import get_logger
import gym
import warnings
import argparse
import yaml
import datetime
import os
from pathlib import Path

warnings.filterwarnings('ignore')

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path


class MergedConfig:
    log_dir: str
    load_path: str
    model_dir: str
    env_name: str
    load_checkpoint: bool
    mode: str
    train_eps: int
    eval_per_episode: int
    test_eps: int
    res_dir: str
    task_dir: str
    device: str
    algo_name: str

    def __init__(self) -> None:
        pass


class Main():

    def __init__(self) -> None:
        pass

    def load_config_from_yaml(self):
        parser = argparse.ArgumentParser(description="hyperparameters")
        parser.add_argument('--yaml',
                            default=None,
                            type=str,
                            help='the path of config file')
        args = parser.parse_args()
        with open(args.yaml) as f:
            load_cfg = yaml.load(f, Loader=yaml.FullLoader)
            algo_name = load_cfg['general_cfg']['algo_name']
            alog_config_mod = __import__(f"models.{algo_name}.config",
                                         fromlist=['AlgoConfig'])
            cfg = {
                "general_cfg": DefaultConfig(),
                "algo_cfg": alog_config_mod.AlgoConfig()
            }
            for cfg_type in cfg:
                if load_cfg[cfg_type] is not None:
                    for k, v in load_cfg[cfg_type].items():
                        setattr(cfg[cfg_type], k, v)
            merged_cfg = MergedConfig()
            merged_cfg = merge_class_attrs(merged_cfg, cfg['general_cfg'])
            merged_cfg = merge_class_attrs(merged_cfg, cfg['algo_cfg'])
            self.cfg = cfg
        return merged_cfg

    def create_path(self, cfg):
        curr_time = datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")  # obtain current time
        task_dir = f"{curr_path}/tasks/{cfg.mode.capitalize()}_{cfg.env_name}_{cfg.algo_name}_{curr_time}"
        setattr(cfg, 'task_dir', task_dir)
        Path(cfg.task_dir).mkdir(parents=True, exist_ok=True)
        model_dir = f"{task_dir}/models"
        setattr(cfg, 'model_dir', model_dir)
        res_dir = f"{task_dir}/results"
        setattr(cfg, 'res_dir', res_dir)
        log_dir = f"{task_dir}/logs"
        setattr(cfg, 'log_dir', log_dir)
        traj_dir = f"{task_dir}/traj"
        setattr(cfg, 'traj_dir', traj_dir)

    def create_env(self, cfg):
        env = gym.make(cfg.env_name, new_step_api=True)
        setattr(cfg, 'n_states',
                env.observation_space.shape[0])  # type: ignore
        setattr(cfg, 'n_actions', env.action_space.n)  # type: ignore
        setattr(cfg, 'action_space', env.action_space)
        return env

    def create_agent(self, cfg):
        agent_mod = __import__(f"models.{cfg.algo_name}.agent",
                               fromlist=['Agent'])
        agent = agent_mod.Agent(cfg)
        return agent

    def create_trainer(self, cfg):
        trainer_mod = __import__(f"models.{cfg.algo_name}.trainer",
                                 fromlist=['Trainer'])
        trainer = trainer_mod.Trainer()
        return trainer

    def run(self):
        cfg = self.load_config_from_yaml()
        self.create_path(cfg)
        logger = get_logger(cfg.log_dir)
        register_env(cfg.env_name)

        env = self.create_env(cfg)
        agent = self.create_agent(cfg)
        trainer = self.create_trainer(cfg)
        if cfg.load_checkpoint:
            agent.load_model(f"tasks/{cfg.load_path}/models")

        if cfg.env_name == "LoadVideo-v0":
            print(
                f"Random reward: {env.get_random_reward()}, Greedy reward: {env.get_greedy_reward()}"  # type: ignore
            )

        rewards = []  # record rewards for all episodes
        steps = []  # record steps for all episodes
        if cfg.mode.lower() == 'train':
            best_ep_reward = -float('inf')
            for i_ep in range(cfg.train_eps):
                agent, ep_reward, ep_step = trainer.train_one_episode(
                    env, agent, cfg)
                logger.info(
                    f"Episode: {i_ep + 1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}"
                )
                if cfg.env_name == "LoadVideo-v0":
                    buffer_map, upsample_map = env.get_action_cnt_map()
                    logger.info(
                        f"Buffer info: {buffer_map}, Upsample info: {upsample_map}"
                    )
                rewards.append(ep_reward)
                steps.append(ep_step)
                # for _ in range
                if (i_ep + 1) % cfg.eval_per_episode == 0:
                    mean_eval_reward = self.evaluate(cfg, trainer, env, agent)
                    if mean_eval_reward >= best_ep_reward:  # update best reward
                        logger.info(
                            f"Current episode {i_ep + 1} has the best eval reward: {mean_eval_reward:.3f}"
                        )
                        best_ep_reward = mean_eval_reward
                        agent.save_model(
                            cfg.model_dir)  # save models with best reward
            # env.close()
        elif cfg.mode.lower() == 'test':
            for i_ep in range(cfg.test_eps):
                agent, ep_reward, ep_step = trainer.test_one_episode(
                    env, agent, cfg)
                logger.info(
                    f"Episode: {i_ep + 1}/{cfg.test_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}"
                )
                rewards.append(ep_reward)
                steps.append(ep_step)
            agent.save_model(cfg.model_dir)  # save models

        res_dic = {
            'episodes': range(len(rewards)),
            'rewards': rewards,
            'steps': steps
        }
        save_results(res_dic, cfg.res_dir)  # save results
        save_cfgs(self.cfg, cfg.task_dir)  # save config
        plot_rewards(
            rewards,
            title=
            f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}",
            fpath=cfg.res_dir)

    def evaluate(self, cfg, trainer, env, agent):
        sum_eval_reward = 0
        for _ in range(cfg.eval_eps):
            _, eval_ep_reward, _ = trainer.test_one_episode(env, agent, cfg)
            sum_eval_reward += eval_ep_reward
        mean_eval_reward = sum_eval_reward / cfg.eval_eps
        return mean_eval_reward


if __name__ == "__main__":
    main = Main()
    main.run()