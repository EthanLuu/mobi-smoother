from gym.envs.registration import register

def register_env(env_name):
    if env_name == 'LoadVideo-v0':
        register(
            id='LoadVideo-v0',
	        entry_point = 'envs.load_video:LoadVideoEnv',
            max_episode_steps=1000,
            kwargs={}
        )
    else:
        print("The env name must be wrong or the environment donot need to register!")

