import random
import os
import numpy as np
from networks import DQNNet, DQNPoly
from replay_buffer import ReplayBuffer
from shared import DISCOUNT, BATCH_SIZE, UPDATE_RATE


class Agent(object):

    def __init__(self, obs_shape, num_actions):
        self.num_actions = num_actions

        self.trainer = DQNNet(obs_shape, self.num_actions, learning_rate=0.001)
        self.target = DQNNet(obs_shape, self.num_actions, learning_rate=0.001)
        self.simple = DQNPoly((11, 16), self.num_actions, learning_rate=0.001)

        self.replay = ReplayBuffer(100000)

        self.prev_state = None
        self.prev_action = None
        self.last_change = 0

        self.ep = 1
        self.explore = 5

    def load(self, env_id):
        self.trainer.load('Data' + env_id + os.sep + 'policy.net')
        self.target.model.set_weights(self.trainer.model.get_weights())
        self.simple.load('Data' + env_id + os.sep + 'poly.fn')
        print('loaded from ' + env_id)

    def episode_end(self, env_id):
        self.trainer.save('Data' + env_id + os.sep + 'policy.net')
        self.simple.save('Data' + env_id + os.sep + 'poly.fn')
        self.prev_action = None
        self.last_change = 0
        self.ep += 1
        if self.ep == 20:
            print('Exploration off')
            self.explore = 0

    def act(self, state, reward, done):
        if self.prev_action is not None:
            prev_all_zero = self.prev_state[1][:, :-4].any()
            if prev_all_zero:  # End of episode sends 0 demand waiting for vehicles to clear map, ignore those steps
                self.replay.add(self.prev_state, self.prev_action, reward, state, done)
                self.last_change += 1

            if len(self.replay) > BATCH_SIZE * 2:
                self.train()
                self.train_simple()

            if self.last_change == UPDATE_RATE:
                self.target.model.set_weights(self.trainer.model.get_weights())
                self.last_change = 0

        predicted = self.simple.values(state[1])
        best_action = np.argmax(predicted)

        if random.uniform(0, 100) < self.explore:
            action = random.choice(range(self.num_actions))
        else:
            action = best_action

        self.prev_state = state
        self.prev_action = action
        return action

    def train(self):
        states, actions, rewards, next_states, _ = self.replay.sample(BATCH_SIZE)
        poly_states, raw_states, raw_next, poly_next = [], [], [], []
        for i in range(BATCH_SIZE):
            raw_states.append(states[i][0])
            poly_states.append(states[i][1])
            raw_next.append(next_states[i][0])
            poly_next.append(next_states[i][1])
        raw_states = np.array(raw_states)
        poly_states = np.array(poly_states)
        raw_next = np.array(raw_next)

        # Discount intermediate no-op state rewards
        future_rewards = self.target.best_value(raw_next)
        total_reward = []
        for i in range(len(rewards)):
            actual_reward = 0
            for j, reward in enumerate(rewards[i]):
                actual_reward += (DISCOUNT ** j) * reward
            actual_reward += (DISCOUNT ** len(rewards[i])) * future_rewards[i]
            total_reward.append(actual_reward)

        exp_fut_rew = self.target.best_value(raw_next)
        total_reward = rewards + DISCOUNT * exp_fut_rew

        self.trainer.train(raw_states, actions, total_reward)
        self.simple.train(poly_states, actions, total_reward)

    def train_simple(self):
        for i in range(10):
            states, actions, rewards, next_states, _ = self.replay.sample(BATCH_SIZE)
            poly_states, raw_states, raw_next = [], [], []
            for i in range(BATCH_SIZE):
                raw_states.append(states[i][0])
                poly_states.append(states[i][1])
                raw_next.append(next_states[i][0])
            raw_next = np.array(raw_next)
            poly_states = np.array(poly_states)

            exp_fut_rew = self.target.best_value(raw_next)
            total_reward = rewards + DISCOUNT * exp_fut_rew

            self.simple.train(poly_states, actions, total_reward)
