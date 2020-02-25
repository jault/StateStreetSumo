import shared
from traci_env.envs.Constants import Phase
import numpy as np
import gym
import cma
import multiprocessing as mp
import time


NUM_ACTIONS = 11
NUM_MEASUREMENTS = 100
CMA_POP = 12
CMA_SIMS = 2
CMA_VARIANCE = 0.2
FILE = 1


def main():
    opts = cma.CMAOptions()
    opts.set('popsize', CMA_POP)
    num_params = 2 * NUM_ACTIONS * NUM_MEASUREMENTS
    lower = [0.00001] * num_params
    upper = [np.inf] * num_params
    opts.set('bounds', [lower, upper])
    initial = [1.] * num_params
    cmaes = cma.CMAEvolutionStrategy(initial, CMA_VARIANCE, opts)

    for ep_cnt in range(200):
        solutions = cmaes.ask()
        results = []
        pool = mp.Pool(processes=24)
        manager = mp.Manager()
        return_dict = manager.dict()

        for m in range(CMA_POP):
            for n in range(CMA_SIMS):
                print('eps'+str(ep_cnt)+'tr'+str(m)+str(n))
                pool.apply_async(runner, args=('eps'+str(ep_cnt)+'tr'+str(m), str(n), solutions[m], return_dict))
                time.sleep(5.0)

        pool.close()
        pool.join()

        for m in range(CMA_POP):
            summed = 0.0
            for n in range(CMA_SIMS):
                summed += return_dict['eps'+str(ep_cnt)+'tr'+str(m) + str(n)]
            results.append(summed / CMA_SIMS)
        print('Completed iteration', ep_cnt, 'perf', sum(results) / CMA_POP)
        cmaes.tell(solutions, results)
        print('Best solution found so far:')
        print(cmaes.result.xbest)


def runner(trial_id, eps_id, params, return_dict):
    env = gym.make('traci-v0')
    env.trial_id = trial_id
    env.eps_id = eps_id
    env.demand_file = shared.demands[FILE]
    env.rush_hour = shared.rush[FILE]
    env.dead_hour = shared.dead[FILE]
    env.open()

    agt = Agent(params, env)
    state = env.get_state()
    reset = False
    while not reset:
        state, rew, reset, _ = env.step(action=agt.act(state, None, None))
    loss = env.close()
    return_dict[trial_id+eps_id] = loss


class Agent(object):

    def __init__(self, parameters, env):
        self.env = env
        self.upper_phase = Phase.E
        self.lower_phase = Phase.W
        self.parameters = parameters

    def act(self, state, reward, done):
        action_values = []
        state = state.flatten()
        state_act = []
        for i in range(NUM_ACTIONS):
            state_act.append(np.copy(state))
        state = np.stack(state_act)
        state = state.flatten()

        w = self.parameters[:len(state)]
        p = self.parameters[len(state):]
        demand = np.power(np.multiply(w, state), p)
        for i in range(len(self.env.actions)):
            sum = 0
            for j in range(NUM_MEASUREMENTS):
                sum += demand[i * NUM_MEASUREMENTS + j]
            action_values.append(sum)

        action = np.argmax(action_values)

        self.upper_phase = self.env.actions[action][0]
        self.lower_phase = self.env.actions[action][1]
        return action

    def episode_end(self, env_id):
        return


if __name__ == "__main__":
    main()
