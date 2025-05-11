# CARLA Reinforcement Learning Project Documentation

## Project Overview

This project implements various reinforcement learning algorithms for autonomous driving using the CARLA simulator. The system trains agents to drive vehicles safely through urban environments, follow lanes, handle junctions, and avoid collisions.

## Repository Structure

```
carla_server_codes/
├── algorithms/              # RL algorithm implementations
│   ├── __init__.py          # Package initialization
│   ├── base_agent.py        # Abstract base class for all agents
│   ├── dqn.py               # Deep Q-Network implementation
│   ├── ddqn.py              # Double Deep Q-Network implementation
│   ├── dueling_dqn.py       # Dueling Deep Q-Network implementation
│   ├── sarsa.py             # SARSA implementation
│   └── ppo.py               # Proximal Policy Optimization implementation
├── carla_env/               # CARLA environment interface
│   ├── __init__.py          # Package initialization
│   ├── carla_client.py      # CARLA client implementation
│   └── rl_env.py            # Gym-compatible RL environment
├── utils/                   # Utility functions and classes
│   ├── replay_buffer.py     # Experience replay buffer
│   └── helpers.py           # Helper functions
├── models/                  # Directory for saved models
├── logs/                    # Directory for TensorBoard logs
├── results/                 # Directory for results and plots
├── main.py                  # Main script for training and evaluation
├── config.py                # Configuration parameters
├── requirements.txt         # Project dependencies
└── README.md                # Project overview
```

## Environment Architecture

### CARLA Environment (`carla_env/rl_env.py`)

The `CarlaEnv` class provides a gym-compatible reinforcement learning environment that interfaces with the CARLA simulator:

1. **Observation Space**: 
   - RGB camera images (84x84x3 by default)

2. **Action Space**: 
   - Discrete action space with 7 actions (defined in `config.py`):
     - No action [0.0, 0.0]
     - Throttle [1.0, 0.0]
     - Left [0.0, -0.5]
     - Right [0.0, 0.5]
     - Throttle + Left [0.5, -0.5]
     - Throttle + Right [0.5, 0.5]
     - Brake [-0.5, 0.0]

3. **Reward System**:
   - Forward Movement: +0.5 for moving forward
   - Collision Penalty: -100.0 for collisions
   - Lane Invasion Penalty: -5.0 for crossing lane markings
   - Centerline Distance Penalty: -0.1 * distance from lane center
   - Speed Reward: +0.2 * (speed/30) for maintaining speed (capped at 30 km/h)
   - Time Penalty: -0.1 per timestep

4. **Termination Conditions**:
   - Collision: Episode ends immediately upon collision
   - Vehicle Stuck: Episode ends if vehicle moves less than 0.5m over 10 timesteps and more than 50 steps have passed
   - Maximum Steps: Episode ends after 1000 steps (configurable)

5. **Episode Statistics**:
   - Tracks rewards, centerline deviations, junction actions, collisions, and lane invasions

### CARLA Client (`carla_env/carla_client.py`)

The `CarlaClient` class handles the direct interaction with the CARLA simulator:

1. **Sensors**:
   - RGB Camera: Front-facing camera providing 84x84 RGB images
   - Collision Sensor: Detects collisions with other objects
   - Lane Invasion Sensor: Detects when the vehicle crosses lane markings

2. **Vehicle Control**:
   - Converts discrete actions to continuous control signals (throttle, steer, brake)
   - Applies control to the vehicle in the simulation

3. **Environment State**:
   - Provides vehicle state (position, velocity, rotation)
   - Calculates distance from the center of the lane
   - Detects when vehicle is at junctions

## Reinforcement Learning Algorithms

The project implements several reinforcement learning algorithms, all inheriting from a common `BaseAgent` class:

### Base Agent (`algorithms/base_agent.py`)

An abstract class providing common functionality for all RL agents:

1. **Common Interface**:
   - `select_action(state, training=True)`: Select an action based on the current state
   - `update(state, action, reward, next_state, done)`: Update the agent's knowledge based on experience
   - `preprocess_state(state)`: Preprocess the state for neural network input

2. **Metrics Tracking**:
   - Episode rewards, lengths, and losses
   - Environment-specific metrics like centerline deviations and junction actions

3. **Utility Methods**:
   - Model saving and loading
   - Metrics logging and visualization

### Deep Q-Network (`algorithms/dqn.py`)

Implements the DQN algorithm with experience replay and target networks:

1. **Neural Network**:
   - Convolutional neural network for processing images
   - Outputs Q-values for each possible action

2. **Learning Process**:
   - Off-policy learning with experience replay
   - Uses epsilon-greedy exploration strategy
   - Periodic target network updates for stability

### Double Deep Q-Network (`algorithms/ddqn.py`)

Extends DQN with the Double DQN improvement:

1. **Action Selection vs. Evaluation**:
   - Uses online network for action selection
   - Uses target network for action evaluation
   - Reduces overestimation of Q-values

### Dueling Deep Q-Network (`algorithms/dueling_dqn.py`)

Extends Double DQN with the dueling architecture:

1. **Value and Advantage Streams**:
   - Separates state value estimation from action advantage estimation
   - Improves learning by focusing on relevant state features

### SARSA (`algorithms/sarsa.py`)

Implements the SARSA (State-Action-Reward-State-Action) algorithm:

1. **On-Policy Learning**:
   - Uses the actual next action to update Q-values (unlike off-policy DQN)
   - More conservative learning approach

### Proximal Policy Optimization (`algorithms/ppo.py`)

Implements the PPO algorithm with the clipped surrogate objective:

1. **Actor-Critic Architecture**:
   - Policy network (actor) outputs action probabilities
   - Value network (critic) estimates state values

2. **Policy Optimization**:
   - Collects trajectories of experiences
   - Updates policy using clipped surrogate objective
   - Multiple optimization epochs per batch of data

## Training Process

The training process is orchestrated by the `train` function in `main.py`:

1. **Environment Initialization**:
   - Creates CARLA environment with specified parameters
   - Sets up the appropriate RL agent

2. **Training Loop**:
   - For each episode:
     - Reset environment and agent
     - Episode interaction loop:
       - Agent selects action based on current state
       - Environment executes action, returns reward and next state
       - Agent stores experience and updates policy
     - Compute episode statistics
     - Save metrics and agent checkpoints
     - Plot learning curves

3. **Performance Metrics**:
   - Episode rewards and losses
   - Centerline deviations
   - Collision and lane invasion counts
   - Junction actions (left, right, forward, other)

## Evaluation Process

The evaluation process is handled by the `evaluate` function in `main.py`:

1. **Model Loading**:
   - Loads a trained model from a checkpoint

2. **Evaluation Loop**:
   - Runs multiple episodes with reduced exploration
   - Collects performance metrics

3. **Results Analysis**:
   - Computes average reward, episode length, and other metrics
   - Generates plots and visualizations

## Configuration

Key configuration parameters in `config.py`:

1. **CARLA Parameters**:
   - Host, port, timeout, and map selection
   - Weather conditions
   - Vehicle type and spawn point

2. **Environment Parameters**:
   - Frame rate and maximum episode steps
   - Sensor configurations
   - Action space definition

3. **Reward Parameters**:
   - Coefficients for different reward components
   - Penalty values for violations

4. **Algorithm Hyperparameters**:
   - Learning rates and discount factors
   - Exploration parameters
   - Network architectures

## Limitations and Future Work

1. **Goal-Directed Navigation**:
   - Current implementation focuses on general driving skills
   - No explicit destination or route planning
   - Target-reaching logic is a placeholder and not implemented

2. **Potential Improvements**:
   - Implement goal-directed navigation with specific destinations
   - Add more sophisticated sensors (LIDAR, depth cameras)
   - Implement attention mechanisms for better feature extraction
   - Add traffic and pedestrian interaction

## Usage Instructions

1. **Training**:
   ```
   python main.py --algorithm [dqn|ddqn|dueling_dqn|sarsa|ppo] --train --carla_host localhost --carla_port 2000 --episodes 1000
   ```

2. **Evaluation**:
   ```
   python main.py --algorithm [dqn|ddqn|dueling_dqn|sarsa|ppo] --eval --load_checkpoint [checkpoint_path] --eval_episodes 10
   ```

3. **Algorithm Comparison**:
   ```
   python main.py --compare --algorithms dqn ddqn dueling_dqn --episodes 500
   ``` 