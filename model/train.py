import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from agent import QLearningAgent
from poker_env import PokerEnvironment

def train(episodes=50000000):
    env = PokerEnvironment()
    # Start with high exploration, decay slowly
    agent = QLearningAgent(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999995) 
    
    rewards_history = []
    win_rate_history = []
    match_wins = 0
    
    print(f"Starting training for {episodes} episodes (BO3 Matches)...")
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, result, done = env.redraw(action)
            
            # Update Q-table
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon after each episode
        agent.decay_epsilon()
        
        rewards_history.append(total_reward)
        
        # Check if player won the match (BO3)
        if env.player_wins > env.opponent_wins:
            match_wins += 1
            
        if (episode + 1) % 10000 == 0:
            win_rate = match_wins / 10000
            win_rate_history.append(win_rate)
            print(f"Episode {episode + 1}/{episodes} - Match Win Rate (last 10k): {win_rate:.2f} - Epsilon: {agent.epsilon:.4f} - Q-Table Size: {len(agent.q_table)}")
            match_wins = 0
            
    agent.save_q_table()
    
    # Plotting results
    plt.plot(win_rate_history)
    plt.title("Win Rate over Training")
    plt.xlabel("Episodes (x1000)")
    plt.ylabel("Win Rate")
    plt.savefig("training_results.png")
    print("Training complete. Results saved to training_results.png")

if __name__ == "__main__":
    train()
