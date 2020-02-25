#!/usr/bin/env python3
import gym
from distutils.dir_util import copy_tree
import xml.etree.ElementTree as ElementTree
import os
import sys
from gym import spaces
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
from sumolib import checkBinary

from . import SimThread
from . import Constants as C
from .Constants import LAD
from .Constants import IND
from .Constants import Phase
from .Constants import WL_EL, E_W, SL_NL, NNL_SSL
import numpy as np

path = os.path.dirname(os.path.abspath(__file__)) + os.sep

OBS_SPACE = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(20, 5, 1))
ACT_SPACE = spaces.Discrete(C.NUM_ACTIONS)


class SimInternal(object):
    def __init__(self, demand):
        self.demand = demand
        self.vehNr = 0
        self.current_phase = [2, 6]
        self.yellow = False
        self.conflict_red = None
        self.barrier_red = False
        self.last_phase_change = 0
        self.acted = False
        self.conflict_red_count = 0
        self.barrier_red_count = 0


class TraciEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.trial_id = 0
        self.eps_id = 0
        self.demand_file = None
        self.rush_hour = None
        self.dead_hour = None
        self.state_mode = None
        self.ew_upper_ring = [Phase.E, Phase.W_L]
        self.ew_lower_ring = [Phase.W, Phase.E_L]
        self.ns_upper_ring = [Phase.NN_L, Phase.N, Phase.S_L]
        self.ns_lower_ring = [Phase.SS_L, Phase.S, Phase.N_L]
        self.actions = self.get_all_actions()
        self.prev_score = None
        self.upper_phase = None
        self.lower_phase = None
        self.demand = None
        self.sim = None
        self.sim_intern = None
        self.env_id = None
        self.sumoBinary = checkBinary('sumo')
        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE
        self.steps = 0

    def open(self):
        self.env_id = str(self.trial_id) + '-' + str(self.eps_id)
        copy_tree(path+'Data', 'Data'+self.env_id)

        with open(path+self.demand_file, 'r') as fp:  # Load the demand file
            demand = fp.readlines()

        demand.pop(0)
        demand.pop(0)  # The two first entries are the file header, dispose them
        self.sim = SimThread.create_sim(self.sumoBinary, self.env_id, '')
        self.sim_intern = SimInternal(demand)
        self.prev_score = 0
        self.upper_phase = Phase.E
        self.lower_phase = Phase.W
        self.steps = 0

    def close(self):
        self.sim.close()
        loss, loss_sq, trips, rh_loss, dh_loss, max_delay = self.get_delay('Data'+self.env_id+os.sep)
        missed_trips = self.sim_intern.vehNr - trips
        if missed_trips > 0:
            print('MISSED TRIPS')
        penalized_loss = loss + missed_trips * C.MAX_WAIT

        max_queue = self.get_max_queue('Data'+self.env_id+os.sep)

        with open("perf.csv", "a+") as f:
            f.write(str(self.trial_id) + ', ' + str(self.eps_id) + ', ' + str(trips) + ', ' + str(missed_trips) + ', ' +
                    str(loss) + ', ' + str(self.steps) + ', ' + str(rh_loss) + ', ' + str(dh_loss) + ', ' +
                    str(max_delay) + ', ' + str(max_queue) + '\n')
        print(self.env_id, 'Completed iteration trips', trips, 'loss', loss, 'steps', self.steps, 'rush', rh_loss, 'dead', dh_loss)
        return penalized_loss

    def step(self, action):
        self.steps += 1
        reset = False
        if self.sim.simulation.getTime() >= 55000:
            reset = True

        actuation = False
        if action >= 100:   # Actuated action
            action -= 100
            phase = self.get_phase_from_action(action)
            actuation = True
        else:
            phase = self.get_phase_from_action(action)

        rewards = []
        self.sim_intern.acted = False
        # Collect rewards from intermediate no-op states
        while not self.sim_intern.acted:
            SimThread.run_sim(self.sim, self.sim_intern, phase, actuation)
            rewards.append(self.get_waiting_reduction())

        self.upper_phase = self.actions[action][0]
        self.lower_phase = self.actions[action][1]
        return self.get_state(), rewards, reset, {}

    def reset(self):
        if self.sim is not None: self.close()
        self.open()
        return self.get_state()

    def render(self, mode='human'):
        self.sumoBinary = checkBinary('sumo-gui')
        return

    def get_state(self):
        if self.state_mode == 'poly':
            return [self.raw_state(), self.poly_state()]
        elif self.state_mode == 'ind':
            return [self.inductor_state(), self.raw_state(), self.poly_state()]
        else:
            return self.raw_state()

    def inductor_state(self):
        current_signals = list(self.sim.trafficlight.getRedYellowGreenState("gneJ1"))
        if current_signals == WL_EL:
            loops = ['W3', 'W4', 'E3', 'E4']
        elif current_signals == E_W:
            loops = ['W0', 'W1', 'W2', 'E0', 'E1', 'E2']
        elif current_signals == SL_NL:
            loops = ['S4', 'N4']
        elif current_signals == NNL_SSL:
            loops = ['N0', 'N1', 'N2', 'N3', 'N4', 'S0', 'S1', 'S2', 'S3', 'S4']
        else:
            return [-1, -1]     # Not configured

        lastDetected = []
        for loop in loops:
            lastDetected.append(self.sim.inductionloop.getTimeSinceDetection('inductionLoop.'+loop))
        return [min(lastDetected), self.sim.simulation.getTime()]

    def raw_state(self):
        current_signals = list(self.sim.trafficlight.getRedYellowGreenState("gneJ1"))

        phase_light_idx = [9, 10, 16, 18, 19, 15, 0, 1, 2, 3, 20, 21, 6, 7, 8, 4, 11, 12, 13, 14]
        state = np.zeros_like(C.INITIAL_STATE)
        lane_ids = []
        for i, phase in enumerate(LAD, 0):
            if i == 8: break  # Don't use NS or EW aggregate LADs
            for lane in phase.value:
                lane_ids.append(lane)

        for i, lane in enumerate(lane_ids, 0):
            if current_signals[phase_light_idx[i]] == 'r':
                state[i][0] = 0
            elif current_signals[phase_light_idx[i]] == 'y':
                state[i][0] = 0.3
            elif current_signals[phase_light_idx[i]] == 's':
                state[i][0] = 0.6
            elif current_signals[phase_light_idx[i]] == 'g':
                state[i][0] = 0.9
            elif current_signals[phase_light_idx[i]] == 'G':
                state[i][0] = 1

            approaching_vehicles, waiting, queue_length, speed = 0, 0, 0, 0
            for vehicle in self.sim.lanearea.getLastStepVehicleIDs(lane):
                veh_wait = self.sim.vehicle.getAccumulatedWaitingTime(vehicle)
                speed += self.sim.vehicle.getSpeed(vehicle)
                waiting += veh_wait
                if veh_wait > 0:
                    queue_length += 1
                else:
                    approaching_vehicles += 1
            total = queue_length + approaching_vehicles
            if total != 0:
                speed /= total

            state[i][1] = queue_length / C.MAX_VEHICLES
            state[i][2] = approaching_vehicles / C.MAX_VEHICLES
            state[i][3] = waiting / C.MAX_VEHICLES / C.MAX_WAIT
            state[i][4] = speed / C.MAX_SPEED
        state = np.expand_dims(state, axis=2)
        return state

    def poly_state(self):
        action_values = []
        for action in self.actions:
            upper, lower = action[0], action[1]
            action_inputs = []
            action_inputs += self.get_demand(LAD[upper.name])
            action_inputs += self.get_demand(LAD[lower.name])
            # Additional 3 variables to express added delays from signal changes
            if self.need_barrier_red(upper):
                # Need an all red to switch across barrier (e.g. NS -> EW)
                action_inputs += [1, 0, 0, 0]
            else:
                if self.need_conflict_red(upper, lower):
                    # Need red on conflicting lane (for protected lefts)
                    action_inputs += [0, 1, 0, 0]
                else:
                    if self.upper_phase == upper and self.lower_phase == lower:
                        # No red light required
                        action_inputs += [0, 0, 1, 0]
                    else:
                        # Red light required
                        action_inputs += [0, 0, 0, 1]
            action_values.append(action_inputs)
        return np.array(action_values)

    def need_conflict_red(self, next_upper, next_lower):
        lefts = [Phase.E_L, Phase.W_L, Phase.N_L, Phase.S_L]
        throughs = [Phase.E, Phase.W, Phase.N, Phase.S]
        if self.upper_phase in lefts:
            if next_upper in throughs:
                return True
        if self.upper_phase in throughs:
            if next_upper in lefts:
                return True
        if self.lower_phase in lefts:
            if next_lower in throughs:
                return True
        if self.lower_phase in throughs:
            if next_lower in lefts:
                return True
        return False

    def need_barrier_red(self, next_upper):
        if self.upper_phase in self.ns_upper_ring and next_upper in self.ew_upper_ring:
            return True
        elif self.upper_phase in self.ew_upper_ring and next_upper in self.ns_upper_ring:
            return True
        else:
            return False

    # Values should be 0-1, only if average waiting time over all lanes exceeds C.MAX_WAIT, then waiting is >1
    def get_demand(self, lad):
        approaching, waiting, queued, speed, avg_wait = 0, 0, 0, 0, 0
        for lane in lad.value:
            for vehicle in self.sim.lanearea.getLastStepVehicleIDs(lane):
                veh_wait = self.sim.vehicle.getAccumulatedWaitingTime(vehicle)
                speed += self.sim.vehicle.getSpeed(vehicle)
                waiting += veh_wait
                if veh_wait > 0:
                    queued += 1
                else:
                    approaching += 1
        queue_length = queued / len(lad.value) / C.MAX_VEHICLES
        total = queued + approaching
        if total != 0:
            speed = speed / total / C.MAX_SPEED

        queued = queued / C.MAX_VEHICLES
        approaching = approaching / C.MAX_VEHICLES
        waiting = waiting / C.MAX_VEHICLES / C.MAX_WAIT

        if queued != 0:
            avg_wait = waiting/queued

        return [queued, approaching, waiting, speed, queue_length, avg_wait]

    def get_phase_from_action(self, action_index):
        if action_index == -1: return -1    # Native actuation
        return self.actions[action_index][0].value + self.actions[action_index][1].value

    def get_all_actions(self):
        actions = []
        action_sets = []
        for upper in self.ew_upper_ring:
            for lower in self.ew_lower_ring:
                set_value = set(upper.value + lower.value)
                if self.check_unique_action(action_sets, set_value):
                    action_sets.append(set_value)
                    actions.append([upper, lower])
        for upper in self.ns_upper_ring:
            for lower in self.ns_lower_ring:
                set_value = set(upper.value + lower.value)
                if self.check_unique_action(action_sets, set_value):
                    action_sets.append(set_value)
                    actions.append([upper, lower])
        print(actions)
        return np.asarray(actions)

    # 1 Duplicate action is created when iterating over permissive left phases, purge them
    def check_unique_action(self, action_sets, new_action_set):
        for action_set in action_sets:
            if new_action_set == action_set:
                return False
        return True

    def get_waiting_reduction(self):
        cutoff, curr_score = 0, 0
        for phase in LAD:
            if cutoff == 8: break  # Use basic 8 detectors

            phase_score = 0
            for lane in phase.value:
                lane_score = 0
                for vehicle in self.sim.lanearea.getLastStepVehicleIDs(lane):
                    lane_score += self.sim.vehicle.getAccumulatedWaitingTime(vehicle)
                phase_score += (lane_score / C.MAX_VEHICLES)
            curr_score += phase_score
            cutoff += 1
        diff = self.prev_score - curr_score
        self.prev_score = curr_score
        return diff

    def get_delay(self, data_path):
        loss = 0.0
        loss_sq = 0.0
        tripinfos = ElementTree.parse(data_path+'tripinfo.xml').getroot().findall('tripinfo')
        trips = len(tripinfos)
        rh_trips, rh_loss, dh_trips, dh_loss = 0, 0, 0, 0
        max_delay = 0
        for tripinfo in tripinfos:
            trip_loss = float(tripinfo.get('timeLoss'))
            if trip_loss > max_delay:
                max_delay = trip_loss
            loss += trip_loss
            loss_sq += trip_loss ** 2

            departed = float(tripinfo.get('depart'))
            if departed >= self.rush_hour and departed <= self.rush_hour+3600:
                rh_loss += trip_loss
                rh_trips += 1
            if departed >= self.dead_hour and departed <= self.dead_hour+3600:
                dh_loss += trip_loss
                dh_trips += 1
        if trips == 0: return 0, 0, 0, 0, 0, 0
        if rh_trips == 0 or dh_trips == 0:
            return loss / trips, loss_sq / trips, trips, 0, 0, max_delay
        return loss / trips, loss_sq / trips, trips, rh_loss / rh_trips, dh_loss / rh_trips, max_delay

    def get_max_queue(self, data_path):
        max_queue = 0
        cutoff = 0
        for phase in LAD:
            if cutoff == 8: break  # Use basic 8 detectors
            for lane in phase.value:
                lad_file = data_path + 'F' + lane + '.xml'
                intervals = ElementTree.parse(lad_file).getroot().findall('interval')
                for interval in intervals:
                    maxJamLengthInVehicles = float(interval.get('maxJamLengthInVehicles'))
                    if maxJamLengthInVehicles > max_queue:
                        max_queue = maxJamLengthInVehicles
            cutoff += 1
        return max_queue
