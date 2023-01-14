from ppo2 import PPO
from common.utils import all_seed, merge_class_attrs
from common.models import ActorSoftmax, Critic
from common.launcher import Launcher
from common.memories import PPOMemory
from config.config import GeneralConfigPPO, AlgoConfigPPO
from envs.register import register_env
import gym
import warnings

warnings.filterwarnings('ignore')


class Main(Launcher):

    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'],
                                                     GeneralConfigPPO())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'],
                                                  AlgoConfigPPO())

    def env_agent_config(self, cfg, logger):
        # create env and agent
        register_env(cfg.env_name)
        env = gym.make(cfg.env_name, new_step_api=True)  # create env
        if cfg.seed != 0:  # set random seed
            all_seed(env, seed=cfg.seed)

        n_states = env.observation_space.shape[0]  # type: ignore
        n_actions = env.action_space.n  # type: ignore
        logger.info(
            f"n_states: {n_states}, n_actions: {n_actions}")  # print info

        # update to cfg paramters
        setattr(cfg, 'n_states', n_states)
        setattr(cfg, 'n_actions', n_actions)
        models = {
            'Actor':
            ActorSoftmax(n_states, n_actions, hidden_dim=cfg.actor_hidden_dim),
            'Critic':
            Critic(n_states, 1, hidden_dim=cfg.critic_hidden_dim)
        }
        memory = PPOMemory(cfg.batch_size)  # replay buffer
        agent = PPO(models, memory, cfg)  # create agent
        if cfg.env_name == "LoadVideo-v0":
            print(
                f"Random reward: {env.get_random_reward()}, Greedy reward: {env.get_greedy_reward()}"  # type: ignore
            )
        return env, agent

    def train_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0  # step per episode
        state = env.reset()
        for _ in range(cfg.max_steps):
            action, prob, val = agent.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_step += 1
            agent.memory.push(state, action, prob, val, reward, terminated)
            if ep_step % cfg.update_fre == 0:
                agent.update()
            state = next_state
            if terminated:
                if cfg.env_name == "LoadVideo-v0":
                    print(env.get_action_cnt_map())
                break
        return agent, ep_reward, ep_step

    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0  # step per episode
        state = env.reset()
        for _ in range(cfg.max_steps):
            action, prob, val = agent.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_step += 1
            state = next_state
            if terminated:
                break
        return agent, ep_reward, ep_step


if __name__ == "__main__":
    main = Main()
    main.run()