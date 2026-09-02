# Near-Accident Driver

**Investigation of Near-Accident Scenarios Using a Hybrid Deep Learning Approach**
Combining Imitation Learning and Reinforcement Learning for robust autonomous driving agents.

Authors: Sourabh S Hegde (PES2UG23CS929), Vatsal Jain (PES2UG23CS681)

## Problem Statement

Training autonomous vehicles to handle rare but critical "edge cases" — near-accident scenarios like sudden swerving or braking — is a major challenge in AI. These events are infrequent in real-world driving data, making them hard for models to learn from. Physical testing of such scenarios is dangerous and impractical, so this project uses a high-fidelity simulator to safely train and evaluate a driving agent on collision-avoidance behavior.

## Approach: Hybrid IL + RL

This project combines two learning paradigms in a two-phase pipeline:

1. **Imitation Learning (IL) — "Warm Start":** A CNN is first trained via behavioral cloning to mimic expert human driving. This gives the agent a strong foundational policy and avoids the RL agent starting from random, inefficient actions.
2. **Reinforcement Learning (RL) — "Mastery":** The pre-trained IL model is fine-tuned using Proximal Policy Optimization (PPO), letting the agent explore the environment, refine its policy, and become more robust to novel, high-risk scenarios than pure imitation allows.

## Project Workflow

1. **Data Collection** (`01_data_collector.py`) — A human operator manually drives in the `highway-env` simulator; each frame and the corresponding keyboard action (Up/Down/Left/Right) is logged.
2. **Data Preprocessing** — Captured frames are cropped to the road and immediate surroundings (`rendered_image[120:250, :]`) and logged to `driving_log.csv`, mapping each image to one of 5 discrete actions.
3. **Imitation Learning** — A CNN is trained to classify the correct action from an input frame, saved as `il_model.keras`.
4. **Reinforcement Learning** — A PPO agent, using the same CNN architecture, is trained for 75,000 timesteps to maximize a custom reward function, saved as `rl_model.zip`.
5. **Evaluation** — The final agent is tested over 20 episodes to measure driving performance.

## Model Architecture (CNN)

- **Input layer:** normalizes pixel values
- **5× Conv2D layers:** feature extractors for lane lines, vehicles, and road texture
- **Flatten layer:** converts 2D feature maps to a 1D vector
- **Dense layers:** decision-making layers correlating features to actions
- **Output layer:** Dense layer, 5 neurons, softmax — outputs a probability distribution over the 5 possible actions

## Reward Function (RL)

The PPO agent's behavior is shaped by a custom reward function:
- **+ High speed reward** — for maintaining speed within a target range
- **+ Right-lane reward** — small bonus for staying in right-hand lanes (stability)
- **− Collision penalty (−2)** — large negative reward on collision, the dominant safety signal

## Results

**Baseline Imitation Learning model**
- Accuracy: **88.10%**
- Weighted F1-score: **0.86**
- *Caveat:* the dataset was imbalanced (~86% "IDLE" actions), so the model excelled at predicting the common case but was hesitant on rarer, critical maneuvers like turning or accelerating. Pure IL alone is insufficient for handling rare events and produces a biased agent.

**Final Reinforcement Learning model (PPO, 75,000 timesteps)**
- Mean reward: **29.87** (std. dev. **9.48**) over 20 evaluation episodes
- RL more than doubled the agent's performance over the IL baseline and taught it more decisive behavior.
- The relatively high standard deviation shows the agent mastered common scenarios but isn't yet fully robust — it still fails in some near-accident situations.

**Takeaway:** The hybrid IL + RL strategy is validated as an effective approach for training autonomous driving agents — IL provides a strong behavioral foundation, and RL builds robustness and mastery on top of it.

## Tech Stack

- **Language:** Python 3.12
- **Deep Learning:** TensorFlow / Keras
- **Reinforcement Learning:** Stable-Baselines3 (PPO)
- **RL Environment:** Gymnasium + [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) — a lightweight, pure-Python driving simulator
- **Data Processing:** OpenCV, Pandas

**Note on simulator choice:** CARLA was initially considered but proved complex to set up. The project pivoted to `highway-env` to focus effort on the core ML pipeline rather than environment configuration, which sped up development.

## Repository Contents

- `src/` — source code (data collection, IL training, RL training/evaluation)
- `ML_project_PES2UG23CS929_PES2UG23CS681_Inference_Report.pdf` — full project report
- `ML_project_PES2UG23CS929_PES2UG23CS681_Presentation.pdf` — project presentation slides

## Getting Started

```bash
git clone https://github.com/SourabhHegde14/near-accident-driver.git
cd near-accident-driver
pip install -r requirements.txt  # tensorflow, stable-baselines3, gymnasium, highway-env, opencv-python, pandas

# 1. Collect expert driving data
python src/01_data_collector.py

# 2. Train the imitation learning model
python src/02_train_il.py

# 3. Fine-tune with reinforcement learning (PPO)
python src/03_train_rl.py

# 4. Evaluate the final agent
python src/04_evaluate.py
```

*(Adjust script names/paths above to match your actual `src/` structure.)*

## Authors

- Sourabh S Hegde — PES2UG23CS929
- Vatsal Jain — PES2UG23CS681

## License

This project is open source and available for educational use.
