# Running CARLA Reinforcement Learning with Python 3.7

This document explains how to run the reinforcement learning project with Python 3.7, which is required for compatibility with CARLA 0.9.15.

## Setup Instructions

1. **Install Python 3.7 Dependencies**
   
   The codebase has been updated to work with Python 3.7. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. **Start CARLA Server**
   
   Navigate to your CARLA directory and start the server:
   ```
   cd path\to\carla_0.9.15
   CarlaUE4.exe -windowed -carla-server
   ```

3. **Running the Project**

   To train a reinforcement learning agent:
   ```
   python main.py --algorithm dqn --train
   ```

   Available algorithms:
   - `dqn`: Deep Q Network
   - `ddqn`: Double Deep Q Network
   - `dueling_dqn`: Dueling Deep Q Network
   - `ppo`: Proximal Policy Optimization
   - `sarsa`: State-Action-Reward-State-Action

   To evaluate a trained model:
   ```
   python main.py --algorithm dqn --eval --eval_episodes 10
   ```

   To compare all trained algorithms:
   ```
   python main.py --compare
   ```

## Troubleshooting

1. **CARLA Version Compatibility**
   
   The codebase was tested with CARLA 0.9.13. If you encounter issues with CARLA 0.9.15, you may need to update the CARLA client API calls. The main differences are usually in sensor configurations and weather parameters.

2. **Package Version Issues**
   
   If you encounter package compatibility issues, you can install specific versions:
   ```
   pip install numpy==1.21.6 matplotlib==3.5.3 pygame==2.1.3
   ```

3. **Memory Issues**
   
   Running CARLA with reinforcement learning can be memory-intensive. If you encounter memory issues:
   - Reduce the resolution of observation images in config.py
   - Reduce replay buffer size in config.py
   - Use a smaller network architecture by modifying the agent implementation

## Performance Optimization

For better training performance:
- Use smaller Town maps (Town01 or Town02)
- Reduce sensor resolution in config.py
- Adjust reward function parameters to stabilize learning
- Reduce the action space if needed by modifying DISCRETE_ACTIONS in config.py 