# src/06_train_rl.py
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import os  # Import the os module

# --- Configuration ---
# NEW SAVE PATH LOGIC: Save to the user's home directory to avoid OneDrive issues
HOME_DIRECTORY = os.path.expanduser("~")
SAVE_DIR = os.path.join(HOME_DIRECTORY, "my_rl_models")
RL_MODEL_PATH = os.path.join(SAVE_DIR, "rl_model.zip")
TOTAL_TIMESTEPS = 20000

def main():
    # NEW: Create the save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"--- Model will be saved to: {RL_MODEL_PATH} ---")

    # --- Create the Environment ---
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140]
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 40,
        "initial_lane_id": None,
        "reward_speed_range": [20, 30],
        "collision_reward": -2,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0,
    }
    
    env = gym.make('highway-v0', config=config)

    print("--- Starting Reinforcement Learning Training ---")
    
    model = PPO("CnnPolicy", env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.8,
                verbose=1,
                tensorboard_log="./ppo_highway_tensorboard/")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    model.save(RL_MODEL_PATH)
    print(f"--- RL model successfully saved to {RL_MODEL_PATH} ---")
    
    env.close()

if __name__ == '__main__':
    main()