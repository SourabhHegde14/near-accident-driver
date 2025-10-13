# src/01_data_collector.py
import gymnasium as gym
import highway_env
import cv2
import csv
import os
import keyboard
import time
import numpy as np

# --- Configuration ---
DATA_DIR = '../data'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'driving_log.csv')
ENV_NAME = 'highway-v0'

def save_data(image_path, action):
    """Saves the image path and action to the CSV file."""
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_path, action])

def main():
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'action'])

    # --- Create the Environment ---
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 1,
            "weights": [0.2989, 0.5870, 0.1140]  # ADDED THIS LINE
        },
        "policy_frequency": 5,
        "screen_width": 600, "screen_height": 300,
    }
    
    env = gym.make(ENV_NAME, config=config, render_mode='rgb_array')
    env.reset()

    frame_id = 0
    done = False
    print("--- Manual Control Enabled ---")
    print("  UP: Faster | DOWN: Slower | LEFT: Lane Left | RIGHT: Lane Right | Q: Quit")
    
    while True:
        env.render()

        action = 1 # Default: IDLE
        if keyboard.is_pressed('left arrow'): action = 0
        elif keyboard.is_pressed('right arrow'): action = 2
        elif keyboard.is_pressed('up arrow'): action = 3
        elif keyboard.is_pressed('down arrow'): action = 4
        
        if keyboard.is_pressed('q'):
            print("Stopping data collection.")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rendered_image = env.render()
        cropped_image = rendered_image[120:250, :] 
        
        img_path_rel = os.path.join('images', f"{frame_id}.png")
        img_path_abs = os.path.join(DATA_DIR, img_path_rel)
        cv2.imwrite(img_path_abs, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        
        save_data(img_path_rel, action)
        frame_id += 1

        if done:
            print("Episode finished. Resetting.")
            env.reset()

        time.sleep(0.1)

    env.close()

if __name__ == '__main__':
    main()