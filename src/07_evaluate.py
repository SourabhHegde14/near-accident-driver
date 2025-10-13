# src/07_evaluate.py
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import numpy as np
import os # Import the os module

# --- Configuration ---
# UPDATED PATH: Load the model from the user's home directory
HOME_DIRECTORY = os.path.expanduser("~")
SAVE_DIR = os.path.join(HOME_DIRECTORY, "my_rl_models")
RL_MODEL_PATH = os.path.join(SAVE_DIR, "rl_model.zip")
NUM_TEST_EPISODES = 20

def main():
    # --- Create the Environment for Evaluation ---
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140]
        },
        "action": {"type": "DiscreteMetaAction"},
        "vehicles_count": 50,
        "collision_reward": -2,
    }
    
    env = gym.make('highway-v0', config=config, render_mode='human')

    print(f"--- Loading model from: {RL_MODEL_PATH} ---")
    model = PPO.load(RL_MODEL_PATH, env=env)

    total_rewards = []
    print(f"--- Running {NUM_TEST_EPISODES} Evaluation Episodes ---")

    for episode in range(NUM_TEST_EPISODES):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    env.close()

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print("\n--- Evaluation Finished ---")
    print(f"Mean reward over {NUM_TEST_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == '__main__':
    main()