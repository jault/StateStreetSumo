
class Agent(object):
    MIN_DUR = 5
    MAX_DUR = 300
    MAX_GAP = 3

    def __init__(self, obs_shape, num_actions):
        self.num_actions = num_actions
        self.curr_act = 100
        self.last_change = 0

    def episode_end(self, env_id):
        self.curr_act = 100
        self.last_change = 0

    def act(self, state, reward, done):
        #return -1  # turns on SUMO automatic control
        if reward != -1 and self.num_actions != 4:
            self.curr_act = reward
            if done == False:
                self.last_change = state[0][1] - 5

        if state[0][1] == -1:
            raise Exception('Actuated can not handle all phases currently.')
        elapsed = state[0][1] - self.last_change
        #print(elapsed, state[0][0])
        if elapsed >= self.MIN_DUR and (state[0][0] >= self.MAX_GAP or elapsed > self.MAX_DUR):
            self.last_change = state[0][1]
            if self.curr_act == 100:
                self.curr_act = 110
            elif self.curr_act == 110:
                self.curr_act = 104
            elif self.curr_act == 104:
                self.curr_act = 103
            elif self.curr_act == 103:
                self.curr_act = 100
        return self.curr_act
