import sys
import time
import numpy as np
from constants import *
from environment import *
from state import State
"""
ref_solution.py

A2 reference solution. Not for release!

COMP3702 2021 Assignment 2 Support Code

Last updated by njc 01/09/22
"""


class StateWrapper:

    def __init__(self, environment, state):
        self.environment = environment
        self.state = state

        widget_cells = [widget_get_occupied_cells(self.environment.widget_types[i], state.widget_centres[i],
                                                  state.widget_orients[i]) for i in range(self.environment.n_widgets)]
        self.tgts_solved = 0
        for tgt in self.environment.target_list:
            # loop over all widgets to find a match
            for i in range(self.environment.n_widgets):
                if tgt in widget_cells[i]:
                    # match found
                    self.tgts_solved += 1
                    break
        if self.tgts_solved == len(self.environment.target_list):
            self.is_solved = True
        else:
            self.is_solved = False


class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment

        # optimisations
        self.outcome_cache = {}
        self.reachable_states = None

        # value iteration
        self.vi_values = None
        self.vi_policy = None
        self.vi_max_delta = None

        # policy iteration
        self.pi_policy = None
        self.pi_policy_array = None
        self.pi_converged = False
        self.state_indices = None
        self.action_indices = None
        self.state_numbers = None
        self.n_states = None
        self.n_actions = None
        self.t_array = None
        self.r1_array = None

        # monte-carlo tree search
        if environment.agent_type == 'mcts':
            self.wrapped_states = {}
            self.q_sa = {}
            self.n_s = {}
            self.n_sa = {}
        else:
            self.wrapped_states = None
            self.q_sa = None
            self.n_s = None
            self.n_sa = None
        self.VISITS_PER_SIM = 1     # was 1
        self.MAX_ROLLOUT_DEPTH = 50    # was 50
        self.TRIALS_PER_ROLLOUT = 1
        self.EXP_BIAS = 1.4 * 5000

    def get_reachable_states(self):
        # build a list of reachable states
        init_state = self.environment.get_init_state()
        frontier = [init_state]
        reachable_states = {init_state}
        while len(frontier) > 0:
            s = frontier.pop()
            for a in ROBOT_ACTIONS:
                next_states = [x[0] for x in self.env_get_all_outcomes(s, a)]  # take next_state only
                for ns in next_states:
                    if ns not in reachable_states:
                        reachable_states.add(ns)
                        frontier.append(ns)
        return list(reachable_states)

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self.vi_values = {}  # mapping from state -> value
        self.vi_policy = {}  # mapping from state -> action
        self.vi_max_delta = float('inf')

        # build a list of reachable states
        self.reachable_states = self.get_reachable_states()

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.vi_max_delta < self.environment.alpha

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        max_delta = 0.0
        # loop over all reachable states
        for s in self.reachable_states:
            # for terminal states, no need to update
            if self.environment.is_solved(s):
                continue

            best_q = -float('inf')
            best_a = None
            # loop over all actions (no actions are invalid here)
            for a in ROBOT_ACTIONS:
                total = 0.0
                # loop over all outcomes
                for s1, r, p in self.env_get_all_outcomes(s, a):
                    total += p * (r + (self.environment.gamma * self.vi_values.get(s1, 0.0)))
                # update best action
                if total > best_q:
                    best_q = total
                    best_a = a
            # update max delta
            if abs(best_q - self.vi_values.get(s, 0.0)) > max_delta:
                max_delta = abs(best_q - self.vi_values.get(s, 0.0))
            # update stored value and policy
            self.vi_values[s] = best_q
            self.vi_policy[s] = best_a

        # store max delta
        self.vi_max_delta = max_delta

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        return self.vi_values.get(state, 0.0)

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.vi_policy.get(state, FORWARD)

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        # build a list of reachable states
        self.reachable_states = self.get_reachable_states()
        self.n_states = len(self.reachable_states)
        self.n_actions = len(ROBOT_ACTIONS)
        self.state_numbers = np.array(range(self.n_states))

        # build map of indexes into flat state array
        self.state_indices = {s: i for i, s in enumerate(self.reachable_states)}

        # build map of indexes into flat action array
        self.action_indices = {a: i for i, a in enumerate(ROBOT_ACTIONS)}

        # initialise policy to always move forward
        self.pi_policy_array = np.ones([len(self.reachable_states)], dtype=np.int64) * self.action_indices[FORWARD]

        # build T array
        self.t_array = np.zeros([self.n_actions, self.n_states, self.n_states])
        for s in self.reachable_states:
            if self.environment.is_solved(s):
                continue
            for a in ROBOT_ACTIONS:
                outcomes = self.env_get_all_outcomes(s, a)
                for s1, _, p in outcomes:
                    self.t_array[(self.action_indices[a], self.state_indices[s], self.state_indices[s1])] += p

        # build R' array
        self.r1_array = np.zeros([self.n_actions, self.n_states])
        for s in self.reachable_states:
            for a in ROBOT_ACTIONS:
                total = 0
                outcomes = self.env_get_all_outcomes(s, a)
                for _, r, p in outcomes:
                    total += r * p
                self.r1_array[(self.action_indices[a], self.state_indices[s])] = total

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.pi_converged

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """

        # ===== policy evaluation =====
        # use numpy 'fancy indexing'
        r1_pi = self.r1_array[self.pi_policy_array, self.state_numbers]
        t_pi = self.t_array[self.pi_policy_array, self.state_numbers]
        pi_values = np.linalg.solve(np.identity(self.n_states) - (self.environment.gamma * t_pi), r1_pi)

        # ===== policy improvement =====
        q_values = np.zeros([self.n_actions, self.n_states])
        for a in range(self.n_actions):
            a_policy = np.array([a for _ in range(self.n_states)])
            # use numpy 'fancy indexing'
            r1_a = self.r1_array[a_policy, self.state_numbers]
            t_a = self.t_array[a_policy, self.state_numbers]
            q_values[a] = r1_a + (self.environment.gamma * np.matmul(t_a, pi_values))

        # update policy and check for convergence
        new_policy = np.argmax(q_values, axis=0)
        if np.array_equal(self.pi_policy_array, new_policy):
            self.pi_converged = True
        self.pi_policy_array = new_policy

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return ROBOT_ACTIONS[self.pi_policy_array[self.state_indices[state]]]

    # === Monte Carlo Tree Search ======================================================================================

    def mcts_initialise(self):
        """
        Initialise any variables required before the start of Monte-Carlo Tree Search.
        """
        # self.q_sa = {}      # (State, action) --> float
        # self.n_s = {}       # State --> int
        # self.n_sa = {}      # (State, action) --> int
        # self.wrapped_states = {}     # State --> StateWrapper
        pass

    def mcts_simulate(self, state: State):
        """
        Perform one simulation of MCTS.
        :param state: the current state
        """
        # disallow revisiting states with one rollout
        visited = {}
        self.__mcts_search(state, 0, visited)

    def __mcts_search(self, state: State, depth, visited):
        # helper method - search recursively
        if state not in self.wrapped_states:
            sw = StateWrapper(self.environment, state)
            self.wrapped_states[state] = sw
        else:
            sw = self.wrapped_states[state]

        # check for non-visit conditions
        if (state in visited and visited[state] > self.VISITS_PER_SIM) or (depth > self.MAX_ROLLOUT_DEPTH):
            # choose the best Q-value if one exists
            best_q = -float('inf')
            best_a = None
            for a in ROBOT_ACTIONS:
                if (state, a) in self.q_sa and self.q_sa[(state, a)] > best_q:
                    best_q = self.q_sa[(state, a)]
                    best_a = a
            if best_a is not None:
                return best_q
            else:
                return self.__mcts_random_rollout(state, self.MAX_ROLLOUT_DEPTH - depth, self.TRIALS_PER_ROLLOUT)
        else:
            if state not in visited:
                visited[state] = 1
            else:
                visited[state] += 1

        # check for terminal state
        if sw.is_solved:
            return 0.0

        # check for leaf node
        if state not in self.n_s:
            # ===== leaf node =====
            self.n_s[state] = 0
            return self.__mcts_random_rollout(state, self.MAX_ROLLOUT_DEPTH - depth, self.TRIALS_PER_ROLLOUT)
        else:
            # ===== not a leaf node =====
            # check if there are unvisited actions
            unvisited = []
            for a in ROBOT_ACTIONS:
                if (state, a) not in self.n_sa:
                    unvisited.append(a)
            if len(unvisited) > 0:
                # if there are unvisited actions, choose one at uniform random
                action = random.choice(unvisited)
            else:
                # otherwise, choose action with best U value
                best_u = -float('inf')
                best_a = None
                for a in ROBOT_ACTIONS:
                    u = self.q_sa.get((state, a), 0.0) + \
                        (self.EXP_BIAS * np.sqrt(np.log(self.n_s.get(state, 0)) / self.n_sa.get((state, a), 1)))
                    if u > best_u:
                        best_u = u
                        best_a = a
                action = best_a

            # update counts
            if (state, action) not in self.n_sa:
                self.n_sa[(state, action)] = 1
            else:
                self.n_sa[(state, action)] += 1
            self.n_s[state] += 1

            # execute action and recurse
            r, new_state = self.environment.perform_action(state, action)
            if new_state not in self.wrapped_states:
                s1w = StateWrapper(self.environment, new_state)
                self.wrapped_states[new_state] = s1w
            else:
                s1w = self.wrapped_states[new_state]

            r += self.__mcts_reward_proxy(sw, s1w)
            r += self.environment.gamma * self.__mcts_search(new_state, depth + 1, visited)

            # update node statistics
            if (state, action) not in self.q_sa:
                self.q_sa[(state, action)] = r
            else:
                self.q_sa[(state, action)] = ((self.q_sa[(state, action)] * self.n_sa[(state, action)]) + r) / \
                                             (self.n_sa[(state, action)] + 1)

            return r

    def __mcts_random_rollout(self, state, max_depth, trials):
        total = 0
        discount = 1.0
        for i in range(trials):
            if state not in self.wrapped_states:
                sw = StateWrapper(self.environment, state)
                self.wrapped_states[state] = sw
            else:
                sw = self.wrapped_states[state]

            d = 0
            while d < max_depth and not sw.is_solved:
                action = random.choice(ROBOT_ACTIONS)
                reward, new_state = self.environment.perform_action(state, action)
                if new_state not in self.wrapped_states:
                    s1w = StateWrapper(self.environment, new_state)
                    self.wrapped_states[new_state] = s1w
                else:
                    s1w = self.wrapped_states[new_state]
                total += discount * (reward + self.__mcts_reward_proxy(sw, s1w))
                state = new_state
                sw = s1w
        return total / trials

    @staticmethod
    def __mcts_reward_proxy(sw: StateWrapper, s1w: StateWrapper):
        # receive proxy reward when moving into the new state (as opposed to once arrived at the new state)
        return sw.tgts_solved - s1w.tgts_solved

    def mcts_select_action(self, state: State):
        """
        Select an approximately optimal action to perform (based on Q-values computed by MCTS).
        :param state: current state
        :return: approximately optimal action to perform for the given state (element of ROBOT_ACTIONS)
        """
        best_q = -np.inf
        best_a = None
        for a in ROBOT_ACTIONS:
            if (state, a) in self.q_sa and self.q_sa[(state, a)] > best_q:
                best_q = self.q_sa[(state, a)]
                best_a = a
        if best_a is not None:
            return best_a
        else:
            return FORWARD

    def mcts_plan_online(self, state: State):
        """
        Plan online using MCTS.
        :param state: current state
        :return: approximately optimal action to perform for the given state (element of ROBOT_ACTIONS)
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        t0 = time.time()
        while time.time() - t0 < self.environment.training_time_tgt:
            self.mcts_simulate(state)
        return self.mcts_select_action(state)

    # === Helper Methods ===============================================================================================

    def env_get_all_outcomes(self, state: State, action):
        if state in self.outcome_cache:
            return self.outcome_cache[(state, action)]

        # outcomes : [(next_state, reward, probability)]
        outcomes = []
        env = self.environment

        # no drift, single move
        p = (1.0 - (env.__drift_cw_probs[action] + env.__drift_ccw_probs[action])) * (1.0 - env.__double_move_probs[action])
        r, s1 = env.__apply_dynamics(state, action)
        outcomes.append((s1, r, p))

        # drift CW, single move
        p = env.__drift_cw_probs[action] * (1.0 - env.__double_move_probs[action])
        r_1, s1 = env.__apply_dynamics(state, SPIN_RIGHT)
        r_2, s1 = env.__apply_dynamics(s1, action)
        r = min(r_1, r_2)
        outcomes.append((s1, r, p))

        # drift CCW, single move
        p = env.__drift_ccw_probs[action] * (1.0 - env.__double_move_probs[action])
        r_1, s1 = env.__apply_dynamics(state, SPIN_LEFT)
        r_2, s1 = env.__apply_dynamics(s1, action)
        r = min(r_1, r_2)
        outcomes.append((s1, r, p))

        # no drift, double move
        p = (1.0 - (env.__drift_cw_probs[action] + env.__drift_ccw_probs[action])) * env.__double_move_probs[action]
        r_1, s1 = env.__apply_dynamics(state, action)
        r_2, s1 = env.__apply_dynamics(s1, action)
        outcomes.append((s1, r, p))

        # drift CW, double move
        p = env.__drift_cw_probs[action] * env.__double_move_probs[action]
        r_1, s1 = env.__apply_dynamics(state, SPIN_RIGHT)
        r_2, s1 = env.__apply_dynamics(s1, action)
        r_3, s1 = env.__apply_dynamics(s1, action)
        r = min(r_1, r_2, r_3)
        outcomes.append((s1, r, p))

        # drift CCW, double move
        p = env.__drift_ccw_probs[action] * env.__double_move_probs[action]
        r_1, s1 = env.__apply_dynamics(state, SPIN_LEFT)
        r_2, s1 = env.__apply_dynamics(s1, action)
        r_3, s1 = env.__apply_dynamics(s1, action)
        r = min(r_1, r_2, r_3)
        outcomes.append((s1, r, p))

        self.outcome_cache[(state, action)] = outcomes
        return outcomes

