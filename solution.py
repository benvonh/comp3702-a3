import sys
import time
import numpy as np
from constants import *
from environment import *
from state import State
"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""


class RLAgent:

    EPISODE = 20000

    def __init__(self, environment: Environment):
        self.environment = environment
        # Set current state
        self.s = environment.get_init_state()
        # Initialise q-table
        self.qtable = { self.s : np.zeros(len(ROBOT_ACTIONS)) }
        # Record start time
        self.start = time.time()

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        for _ in range(self.EPISODE):
            self.s = self.environment.get_init_state()
            while not self.environment.is_solved(self.s):
                # Update action to perform
                action = np.argmax(self.qtable[self.s])
                # Perform action from current state
                cost, next_state = self.environment.perform_action(self.s, action)
                # Add nex discovered
                if next_state not in self.qtable:
                    self.qtable[next_state] = np.zeros(len(ROBOT_ACTIONS))
                # Find max Q in resulting state
                qmax = np.max(self.qtable[next_state])
                # Compute Bellman equation
                self.qtable[self.s][action] += \
                    self.environment.alpha * (cost + self.environment.gamma * qmax - self.qtable[self.s][action])
                # Set next state
                self.s = next_state

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        return np.argmax(self.qtable[state])

    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        for _ in range(self.EPISODE):
            self.s = self.environment.get_init_state()
            # Update action to perform
            action = np.argmax(self.qtable[self.s])
            while not self.environment.is_solved(self.s):
                # Perform action from current state
                cost, next_state = self.environment.perform_action(self.s, action)
                # Add nex discovered
                if next_state not in self.qtable:
                    self.qtable[next_state] = np.zeros(len(ROBOT_ACTIONS))
                # Find epsilon-greedy policy
                action_greedy = np.argmax(self.qtable[next_state])
                # Compute Bellman equation
                self.qtable[self.s][action] += self.environment.alpha * \
                    (cost + self.environment.gamma * self.qtable[next_state][action_greedy] - \
                        self.qtable[self.s][action])
                # Set next state and action
                self.s = next_state
                action = action_greedy

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        return np.argmax(self.qtable[state])

    # === Helper Methods ===============================================================================================
