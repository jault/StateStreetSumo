from gym.envs.registration import register

register(
    id='traci-v0',
    entry_point='traci_env.envs:TraciEnv',
)