import sys
import os
import pickle
import numpy as np
from model_lib.agent import QLearningAgent


Q_TABLE_PATH = os.path.join("model_lib", 'q_table.pkl')

# Global singleton instance
agent_instance = None

def get_agent():
    """
    Get the singleton agent instance. 
    Loads from disk if not already loaded.
    """
    global agent_instance
    
    if agent_instance is not None:
        return agent_instance

    if QLearningAgent is None:
        return None

    print(f"Loading Agent from {Q_TABLE_PATH}...")
    
    agent_instance = QLearningAgent(q_table_path=Q_TABLE_PATH)
    # Force greedy mode for inference
    agent_instance.epsilon = 0.0
      
    return agent_instance

def reload_agent():
    """
    Force reload the agent from disk.
    After the daily training job updates 'q_table.pkl'.
    """
    global agent_instance
    print("Reloading agent from disk...")
    agent_instance = None # Clear the cache
    return get_agent()    # Re-load from file