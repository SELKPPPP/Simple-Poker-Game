import numpy as np
import pickle
import os
import random
from typing import List, Tuple

class QLearningAgent:
    """
    Q-Learning Agent for the Poker Game.
    """
    def __init__(self, n_actions=32, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, q_table_path="q_table.pkl") -> None:
        """
        Initialize the Q-Learning Agent.

        Args:
            n_actions: Number of possible actions (32 for 5 cards redraw decision).
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Initial exploration rate (start high, e.g. 1.0).
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Decay factor per episode.
            q_table_path: Path to save/load the Q-table.
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table_path = q_table_path
        self.q_table = {} # Dictionary mapping state (tuple) to np.array of Q-values

        self.load_q_table()

    def decay_epsilon(self):
        """Decay the exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def choose_action(self, state: tuple, training: bool = True) -> int:
        """
        Choose an action based on the current state using epsilon-greedy policy.

        Args:
            hand: Current hand (list of 5 ints).
            training: Whether we are training (use epsilon) or playing (greedy).

        Returns:
            action: Integer representing the action (0-31).
                    Binary representation indicates which cards to redraw.
                    e.g., 3 (00011) -> Redraw card at index 0 and 1.
                    Define: bit i=1 means REDRAW card i.
        """

        # Epsilon-greedy action selection

        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        if state not in self.q_table:
            # Initialize Q-values for new state (e.g., to 0)
            self.q_table[state] = np.zeros(self.n_actions)
        
        return np.argmax(self.q_table[state])

    def update(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        """
        Update Q-value using the Q-learning update rule.

        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s,a)]

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
        """

        # Ensure states exist in Q-table
        # Initialize Q-values for new states if not present

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)

        # Q-learning update

        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def save_q_table(self):
        """Save the Q-table to a file."""
        with open(self.q_table_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {self.q_table_path}")
        #TODO: Save to AWS 

    def load_q_table(self):
        """Load the Q-table from a file if it exists."""
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {self.q_table_path}, size: {len(self.q_table)}")
        else:
            print("No existing Q-table found. Starting fresh.")
        #TODO: Load from AWS    

    def _indices_to_action(self, indices: List[int]) -> int:
            """
            Convert list of indices to redraw into action index (0-31).
            
            Args:
                indices: List of card indices (0-4) to redraw.
                
            Returns:
                action: Int 0-31
            """
            action = 0
            for idx in indices:
                action |= (1 << idx)
            return action
    
    def _actions_to_indices(self, action: int) -> List[int]:
        """
        Convert action index (0-31) to a list of indices to redraw.

        eg: action 13 (binary 01101) -> redraw indices [0, 2, 3]
        """
        redraw_indices = []
        for i in range(5):
            if (action >> i) & 1:
                redraw_indices.append(i)
        return redraw_indices   
