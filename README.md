# Reinforcement Learning for Autonomous Driving with CARLA

This project implements and compares five different reinforcement learning algorithms for autonomous driving using the CARLA simulator.

## Algorithms Implemented
1. Deep Q Network (DQN)
2. Double Deep Q Network (DDQN)
3. Dueling Deep Q Network
4. Proximal Policy Optimization (PPO)
5. State Action Reward State Action (SARSA)

## Comparison Metrics
1. Average reward per episode
2. Training loss
3. Deviation from the center line
4. Action distribution at junctions

## Project Structure
```
.
├── algorithms/          # RL algorithm implementations
│   ├── base_agent.py    # Base class for all agents
│   ├── dqn.py           # DQN implementation
│   ├── ddqn.py          # Double DQN implementation
│   ├── dueling_dqn.py   # Dueling DQN implementation
│   ├── ppo.py           # PPO implementation
│   └── sarsa.py         # SARSA implementation
├── carla_env/           # CARLA environment integration
│   ├── carla_client.py  # Client for interacting with CARLA
│   └── rl_env.py        # OpenAI Gym environment wrapper
├── utils/               # Utility functions
│   ├── helpers.py       # Helper functions
│   ├── plotting.py      # Plotting utilities
│   └── replay_buffer.py # Replay buffer implementations
├── logs/                # TensorBoard logs
├── models/              # Saved model checkpoints
├── results/             # Training results and plots
├── config.py            # Configuration parameters
├── main.py              # Main script for training and evaluation
└── requirements.txt     # Python dependencies
```

## Setup Instructions

### Prerequisites
- CARLA Simulator (version 0.9.13)
- Python 3.8+
- GPU with CUDA support (recommended)

### Installation
1. Clone this repository
```
git clone https://github.com/yourusername/carla-rl-comparison.git
cd carla-rl-comparison
```

2. Install the required dependencies
```
pip install -r requirements.txt
```

3. Ensure CARLA server is running on the default port (2000)
```
cd /path/to/carla
./CarlaUE4.exe -windowed -carla-server
```

### Running the Project

1. To train a specific algorithm:
```
python main.py --algorithm dqn --train
```

You can specify the following optional arguments:
- `--episodes`: Number of training episodes (default: from config)
- `--seed`: Random seed for reproducibility (default: 42)
- `--model_dir`: Directory to save models (default: 'models')
- `--results_dir`: Directory to save results (default: 'results')
- `--carla_host`: CARLA server host (default: localhost)
- `--carla_port`: CARLA server port (default: 2000)
- `--carla_map`: CARLA map to use (default: Town01)
- `--load_checkpoint`: Path to checkpoint to load
- `--save_interval`: Save model every N episodes

2. To evaluate a trained model:
```
python main.py --algorithm dqn --eval --eval_episodes 10
```

3. To compare all algorithms after training:
```
python main.py --compare
```

This will generate comparative plots for all the trained algorithms.

## Algorithm Details

### 1. Deep Q Network (DQN)
- Uses a target network to stabilize learning
- Implements experience replay to break correlations in the sequence of observations
- Uses epsilon-greedy exploration

### 2. Double Deep Q Network (DDQN)
- Extends DQN to reduce overestimation of Q-values
- Uses separate networks for action selection and evaluation

### 3. Dueling Deep Q Network
- Extends DDQN with a dueling architecture
- Separately estimates state value and action advantages
- Better policy evaluation in states where actions don't affect much

### 4. Proximal Policy Optimization (PPO)
- On-policy algorithm with clipped surrogate objective
- Limits the policy update to prevent too large changes
- Balances exploration and exploitation through entropy bonus

### 5. State-Action-Reward-State-Action (SARSA)
- On-policy temporal difference learning algorithm
- Updates based on the action actually taken in the next state
- Tends to learn more conservative policies

## Configuration
Edit the `config.py` file to adjust hyperparameters, environment settings, and other configurations. Key parameters include:

- CARLA connection details (host, port, map)
- Environment parameters (FPS, sensors, vehicle type)
- Discrete action space definition
- Reward function parameters
- Algorithm-specific hyperparameters

## Results
After training, results and plots will be saved in the `results/` directory. The following plots are generated:

1. Average reward per episode
2. Training loss
3. Centerline deviation
4. Action distribution at junctions
5. Summary plot comparing all metrics across algorithms

## License
MIT 