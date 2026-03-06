# EE547-project

Machine Learning component for the Poker Game Agent.

## Structure

- `agent.py`: Contains the `QLearningAgent` class which implements the Q-Learning algorithm. It handles state representation, action selection (epsilon-greedy), and Q-table updates.
- `poker_env.py`: A simulation environment for the poker game. 
- `poker_rules.py`: Basic rules for poker game
- `train.py`: The main training script. It runs the training loop, letting the agent play against a basic opponent (or baseline) to learn the optimal redraw strategy.
- `q_table.pkl`: (Generated) The serialized Q-table after training.


## State Representation
The state is represented as a tuple containing the following features to abstract the game situation efficiently:
1. **Hand Rank** (0-9): The category of the current hand (e.g., High Card, Pair, Flush).
2. **Top Card** (0-12): The rank of the most significant card in the hand.
3. **Suited Mask** (Tuple of 5): Binary mask indicating cards that are part of a potential flush (>=3 cards of same suit).
4. **Straight Mask** (Tuple of 5): Binary mask indicating cards that are part of a potential straight.
5. **Remaining Redraws**: Number of redraws allowed left in the current round.
6. **Player Wins**: Current match score for the agent.
7. **Opponent Wins**: Current match score for the opponent.


## Action Space
The action is an integer from 0 to 31, representing a bitmask of which cards to redraw (5 cards).
- 0 (00000): Keep all cards.
- 31 (11111): Redraw all cards.

