import random
import os
import numpy as np
from networks import DQNNet, DQNPolyMax
from replay_buffer import ReplayBuffer
from keras.utils import to_categorical
import tkinter as tk
from shared import DISCOUNT, BATCH_SIZE, UPDATE_RATE

avalues = tk.Tk()
rvalues = tk.Tk()

awidth = 800
aheight = 300
rwidth = 1050  # Define it's width
rheight = 200  # Define it's height
avalues.title("Action Values")
acanv = tk.Canvas(avalues, width=awidth, height=aheight, bg='white')
acanv.pack()
rvalues.title("Regulatable Values")
rcanv = tk.Canvas(rvalues, width=rwidth, height=rheight, bg='white')
rcanv.pack()


# The variables below size the bar graph
y_stretch = 15  # The highest y = max_data_value * y_stretch
y_gap = 20  # The gap between lower canvas edge and x axis
x_stretch = 40  # Stretch x wide enough to fit the variables
x_width = 20  # The width of the x-axis
x_gap = 20  # The gap between left canvas edge and y axis


class Agent(object):

    def __init__(self, obs_shape, num_actions, soft):
        self.soft = soft
        self.num_actions = num_actions

        self.trainer = DQNNet(obs_shape, self.num_actions, learning_rate=0.001)
        self.target = DQNNet(obs_shape, self.num_actions, learning_rate=0.001)
        self.simple = DQNPolyMax((11, 16), self.num_actions, learning_rate=0.001)

        self.replay = ReplayBuffer(100000)

        self.prev_state = None
        self.prev_action = None
        self.last_change = 0

        self.ep = 1
        self.explore = 5
        self.count = 1

    def load(self, env_id):
        self.trainer.load('Data'+env_id+os.sep+'policy.net')
        self.target.model.set_weights(self.trainer.model.get_weights())
        self.simple.load('Data' + env_id + os.sep + 'poly.fn')
        print('loaded from ' + env_id)

    def episode_end(self, env_id):
        self.count = 1
        self.trainer.save('Data' + env_id + os.sep + 'policy.net')
        self.simple.save('Data' + env_id + os.sep + 'poly.fn')
        self.prev_action = None
        self.last_change = 0
        self.ep += 1
        if self.ep == 20:
            print('Exploration off')
            self.explore = 0

    def act(self, state, reward, done):
        self.count += 1
        if self.prev_action is not None:
            prev_all_zero = self.prev_state[1][:, :-4].any()
            if prev_all_zero: # End of episode sends 0 demand waiting for vehicles to clear map, ignore those steps
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

        #self.display(state, predicted, best_action, self.simple.model.layers[1].get_weights())

        self.prev_state = state
        self.prev_action = action
        return action

    def train(self):
        states, actions, rewards, next_states, _ = self.replay.sample(BATCH_SIZE)
        poly_states, raw_states, raw_next = [], [], []
        for i in range(BATCH_SIZE):
            raw_states.append(states[i][0])
            poly_states.append(states[i][1])
            raw_next.append(next_states[i][0])
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

        self.trainer.train(raw_states, actions, total_reward)

        if self.soft:
            target_acts = self.trainer.softmax_values(raw_states)
        else:
            target_acts = self.trainer.best_actions(np.array(raw_states))
            target_acts = to_categorical(target_acts, self.num_actions)

        self.simple.train(poly_states, target_acts)

    def train_simple(self):
        for i in range(10):
            states, actions, rewards, next_states, _ = self.replay.sample(BATCH_SIZE)
            poly_states, raw_states, raw_next = [], [], []
            for i in range(BATCH_SIZE):
                raw_states.append(states[i][0])
                poly_states.append(states[i][1])
                raw_next.append(next_states[i][0])
            raw_states = np.array(raw_states)
            poly_states = np.array(poly_states)

            if self.soft:
                target_acts = self.trainer.softmax_values(raw_states)
            else:
                target_acts = self.trainer.best_actions(np.array(raw_states))
                target_acts = to_categorical(target_acts, self.num_actions)
            self.simple.train(poly_states, target_acts)

    def display(self, state, predicted, action, weights):
        acanv.delete("all")
        rcanv.delete("all")
        avalues.attributes('-topmost', True)
        rvalues.attributes('-topmost', True)
        inputs = ['Phase', '(Queue', 'Approach', 'Wait', 'Speed', 'QLength', 'Avg Wait', 'Queue', 'Approach', 'Wait', 'Speed', 'QLength', 'Avg Wait)', '{FULL', 'PART', 'NO', 'PERM}']
        phases = ['E/W', 'E/EL', 'W/WL', 'EL/WL', 'NPL/SPL', 'NPL/S', 'N/NL', 'N/SPL', 'N/S', 'S/SL', 'NL/SL']
        simplebest = np.argmax(predicted)
        nnpred = self.trainer.model.predict(np.asarray([state[0]]))[0]
        nnbest = np.argmax(nnpred)
        # Regulatable Values window
        for x, y in enumerate(inputs):
            x0 = x * x_stretch + x * x_width + x_gap
            rcanv.create_text(x0 + 10, 20, anchor=tk.SW, text=str(y))
        for i in range(len(predicted)):
            if i % 2 == 0:
                rcanv.create_rectangle(x_stretch+40, i * y_stretch + 40, rwidth, i * y_stretch + 25, fill="gray92")
            maxx0 = 0
            y0 = 0
            sum = 0
            for x, y in enumerate(inputs):
                x0 = x * x_stretch + x * x_width + x_gap
                if x0 > maxx0:
                    maxx0 = x0
                y0 = i * y_stretch + 40
                if x == 0:
                    if i == simplebest:
                        rcanv.create_text(x0 + 10, y0, anchor=tk.SW, text=phases[i], fill='red')
                    else:
                        rcanv.create_text(x0 + 10, y0, anchor=tk.SW, text=phases[i])
                else:
                    value = np.power(np.multiply(weights[0][i][x-1], state[1][i][x-1]), weights[1][i][x-1])
                    if x <= 12:
                        sum += value
                    else:
                        if value != 0:
                            sum = sum * value
                    if i == simplebest:
                        rcanv.create_text(x0 + 10, y0, anchor=tk.SW, text=str(np.round(value, 2)), fill='red')
                    else:
                        rcanv.create_text(x0 + 10, y0, anchor=tk.SW, text=str(np.round(value, 2)))
            if i == simplebest:
                rcanv.create_text(maxx0 + 40, y0, anchor=tk.SW, text=str(np.round(sum, 2)), fill='red')
            else:
                rcanv.create_text(maxx0 + 40, y0, anchor=tk.SW, text=str(np.round(sum, 2)))
        # Action values window
        maxx1 = 0
        for x, y in enumerate(predicted):
            x0 = x * x_stretch + x * x_width + x_gap
            y0 = aheight - (y * 10 * y_stretch + y_gap)
            x1 = x * x_stretch + x * x_width + x_width + x_gap
            if x1 > maxx1:
                maxx1 = x1
            y1 = aheight - y_gap
            acanv.create_rectangle(x0, y0, x1, y1, fill="green")
            if x == action:
                acanv.create_text(x0 + 10, 20, anchor=tk.SW, text=phases[x], fill="red")
            elif x == simplebest:
                acanv.create_text(x0 + 10, 20, anchor=tk.SW, text=phases[x], fill="green")
            else:
                acanv.create_text(x0 + 10, 20, anchor=tk.SW, text=phases[x])
        acanv.create_text(maxx1 + 40, 20, anchor=tk.SW, text=str('Avg Reward'))
        for x, y in enumerate(nnpred):
            x0 = x * x_stretch + x * x_width + x_gap + 20
            y0 = aheight - (y * y_stretch + y_gap)
            x1 = x * x_stretch + x * x_width + x_width + x_gap + 20

            y1 = aheight - y_gap
            acanv.create_rectangle(x0, y0, x1, y1, fill="blue")
            if x == nnbest:
                acanv.create_text(x0 - 5, 35, anchor=tk.SW, text=str(np.round(y, 2)), fill="red")
            else:
                acanv.create_text(x0 - 5, 35, anchor=tk.SW, text=str(np.round(y, 2)))
        avalues.update()
