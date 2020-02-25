import os
import numpy as np
from networks import CriticNet, ActorPoly
from shared import DISCOUNT, BATCH_SIZE


ACTOR_STEPS = 128
EPOCHS = 4


class Agent(object):

    def __init__(self, obs_shape, num_actions):
        np.set_printoptions(suppress=True)
        self.num_actions = num_actions

        self.actor = ActorPoly((11, 16), self.num_actions, learning_rate=0.001)
        self.critic = CriticNet(obs_shape, self.num_actions, learning_rate=0.001)

        # prev_state, action, probs, reward, next_state, prev_poly_state, next_poly_state
        self.memory = [[], [], [], [], [], [], []]

        self.prev_state = None
        self.prev_action = None
        self.prev_prob = None

    def load(self, env_id):
        self.actor.load('Data'+env_id+os.sep+'poly.fn')
        self.critic.load('Data' + env_id + os.sep + 'critic.net')
        print('loaded from ' + env_id)

    def episode_end(self, env_id):
        self.actor.save('Data' + env_id + os.sep + 'poly.fn')
        self.critic.save('Data' + env_id + os.sep + 'critic.net')
        self.prev_action = None

    def act(self, state, reward, done):
        raw_state = state[0]
        poly_state = state[1]
        if self.prev_action is not None:
            prev_raw_state = self.prev_state[0]
            prev_poly_state = self.prev_state[1]
            self.memory[0].append(prev_raw_state)
            self.memory[1].append(self.prev_action)
            self.memory[2].append(self.prev_prob)
            self.memory[3].append(reward)
            self.memory[4].append(raw_state)
            self.memory[5].append(prev_poly_state)
            self.memory[6].append(poly_state)

            self.train_critic()
            self.train_actor()

        action, prob = self.actor.get_action([np.zeros((1, 1)), np.zeros((1, self.num_actions)), np.array([poly_state])])

        self.prev_state = state
        self.prev_action = action
        self.prev_prob = prob
        return action

    def train_critic(self):
        if len(self.memory[0]) < ACTOR_STEPS: return

        states = np.array(self.memory[0])
        rewards = np.array(self.memory[3])
        next_states = np.array(self.memory[4])

        # Discount intermediate no-op state rewards
        future_rewards = self.critic.get_state_values(next_states)
        q_val = []
        for i in range(len(rewards)):
            actual_reward = 0
            for j, reward in enumerate(rewards[i]):
                actual_reward += (DISCOUNT ** j) * reward
            actual_reward += (DISCOUNT ** len(rewards[i])) * future_rewards[i]
            q_val.append(actual_reward)

        baseline = self.critic.get_state_values(states)
        advantage = np.asarray(q_val) - baseline

        self.critic.train(states, advantage)

    def train_actor(self):
        if len(self.memory[0]) < ACTOR_STEPS: return

        states = np.array(self.memory[0])
        actions = np.array(self.memory[1])
        probs = np.array(self.memory[2])
        rewards = np.array(self.memory[3])
        next_states = np.array(self.memory[4])
        poly_states = np.array(self.memory[5])

        # Discount intermediate no-op state rewards
        future_val = self.critic.get_state_values(next_states)
        v_state = self.critic.get_state_values(states)
        advantages = []
        for i in range(len(rewards)):
            actual_reward = 0
            for j, reward in enumerate(rewards[i]):
                actual_reward += (DISCOUNT ** j) * reward
            actual_reward += (DISCOUNT ** len(rewards[i])) * future_val[i] - v_state[i]
            advantages.append(actual_reward)

        self.actor.train(poly_states, actions, advantages, probs, BATCH_SIZE, EPOCHS)
        self.memory = [[], [], [], [], [], [], []]
