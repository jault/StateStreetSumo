import shared
import numpy as np
import optparse
import gym
import multiprocessing as mp
from actuated import Agent as ActuatedAgent
from dqn import Agent as DQNAgent
from drq import Agent as DRQAgent
from drxq import Agent as DRXQAgent
from rppo import Agent as RPPOAgent


def main():
    optParser = optparse.OptionParser()
    optParser.add_option("--agent", action="store", default='actu', help="actu, dqn, dqnpq, drsq, drhq, rppo")
    optParser.add_option("--file", action="store", default='0', type='int', help="demand file index (0-2)")
    optParser.add_option("--trials", action="store", default='30', type='int', help="number of trials")
    optParser.add_option("--eps", action="store", default='40', type='int', help="number of episodes per trial")
    optParser.add_option("--procs", action="store", default='30', type='int', help="number of processors to use")
    options, args = optParser.parse_args()
    print(options)

    if options.agent == 'actu':
        num_eps = 1
        num_trials = 30
    else:
        num_eps = options.eps
        num_trials = options.trials

    if options.procs == 1:
        run_trial(options.agent, options.file, num_eps, 0, render=True)
    else:
        pool = mp.Pool(processes=options.procs)
        for trial in range(num_trials):
            pool.apply_async(run_trial, args=(options.agent, options.file, num_eps, trial))
        pool.close()
        pool.join()


def run_trial(agent_type, file, num_eps, trial, render=False):
    mode = 'raw'
    if agent_type == 'actu':
        agent = ActuatedAgent(shared.OBS_SPACE, 4)
    elif agent_type == 'dqn':
        agent = DQNAgent(shared.OBS_SPACE, shared.ACT_SPACE)
    elif agent_type == 'dqnpq':
        agent = DRQAgent(shared.OBS_SPACE, shared.ACT_SPACE)
        mode = 'poly'
    elif agent_type == 'drsq':
        agent = DRXQAgent(shared.OBS_SPACE, shared.ACT_SPACE, soft=True)
        mode = 'poly'
    elif agent_type == 'drhq':
        agent = DRXQAgent(shared.OBS_SPACE, shared.ACT_SPACE, soft=False)
        #agent.load('29-39')
        mode = 'poly'
    elif agent_type == 'rppo':
        agent = RPPOAgent(shared.OBS_SPACE, shared.ACT_SPACE)
        mode = 'poly'
    else:
        raise ValueError('Invalid agent type')

    for ep_cnt in range(num_eps):
        env = gym.make('traci-v0')
        env.state_mode = mode
        env.trial_id = trial
        env.eps_id = ep_cnt
        env.demand_file = shared.demands[file]
        env.rush_hour = shared.rush[file]
        env.dead_hour = shared.dead[file]
        if render: env.render()

        state, rew, reset = env.reset(), 0, False
        while not reset:
            state, rew, reset, _ = env.step(action=agent.act(state, rew, reset))
        agent.episode_end(env.env_id)
        env.close()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
