# Tom & Jerry Reinforcement Learning Agentic RL

This project implements a grid-based reinforcement learning game where an agent (Tom) learns to catch a moving target (Jerry) while avoiding obstacles. The game uses Q-learning algorithm and is visualized using Tkinter.

## Table of Contents
1. [Overview](#overview)
2. [Components](#components)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Running the Application](#running-the-application)
6. [How It Works](#how-it-works)
7. [Customization](#customization)

## Overview

In this game, Tom (the agent) learns to navigate a grid world to catch Jerry (the goal) while avoiding obstacles. The environment is dynamic, with Jerry moving randomly after each of Tom's moves. The game uses reinforcement learning, specifically Q-learning, to train Tom to make optimal decisions.

## Components

The project consists of several key components:

1. `GridWorldMDP`: Implements the Markov Decision Process for the grid world environment.
2. `QLearningAgent`: Implements the Q-learning algorithm for the agent (Tom).
3. `RLVisualizer`: Provides a graphical interface for the game using Tkinter.

## Requirements

- Python 3.x
- NumPy
- Tkinter
- Pillow (PIL)

## Setup

1. Ensure you have Python 3.x installed on your system.
2. Install the required packages:
   ```
   pip install numpy pillow
   ```
3. Tkinter usually comes pre-installed with Python. If it's not available, you may need to install it separately depending on your operating system.

4. Place the following image files in the same directory as the script:
   - `catpic.png` (for Tom)
   - `mousepic.png` (for Jerry)
   - `dead_state.png` (for obstacles)

## Running the Application

To run the application:

1. Open a terminal or command prompt.
2. Navigate to the directory containing the script.
3. Run the following command:
   ```
   python tom_and_jerry_rl.py
   ```

## How It Works

1. The program first trains the Q-learning agent for 5000 episodes.
2. After training, it opens a Tkinter window to visualize the game.
3. Tom (cat) tries to catch Jerry (mouse) while avoiding obstacles.
4. The game resets once Tom catches Jerry or after 100 steps.
5. This process repeats indefinitely until you close the window.

## Customization

You can customize various aspects of the game:

- Adjust the `grid_size` and `obstacle_prob` in the `GridWorldMDP` initialization to change the environment.
- Modify learning parameters (`alpha`, `gamma`, `epsilon`) in the `QLearningAgent` initialization.
- Change the number of training episodes by modifying the `num_episodes` parameter in the `q_agent.train()` call.

Feel free to experiment with these parameters to see how they affect the agent's performance!
