# CARLA Reinforcement Learning Commands

This document contains all the necessary commands for training, evaluating, and comparing different reinforcement learning algorithms in the CARLA environment.

## Training Individual Algorithms

### Training DQN
```bash
python main.py --algorithm dqn --train --carla_host localhost --carla_port 2000 --episodes 1000
```

### Training DDQN
```bash
python main.py --algorithm ddqn --train --carla_host localhost --carla_port 2000 --episodes 1000
```

### Training Dueling DQN
```bash
python main.py --algorithm dueling_dqn --train --carla_host localhost --carla_port 2000 --episodes 1000
```

### Training SARSA
```bash
python main.py --algorithm sarsa --train --carla_host localhost --carla_port 2000 --episodes 1000
```

### Training PPO
```bash
python main.py --algorithm ppo --train --carla_host localhost --carla_port 2000 --episodes 1000
```

## Evaluating Trained Models

### Evaluating DQN
```bash
okay 
```

### Evaluating DDQN
```bash
python main.py --algorithm ddqn --eval --eval_episodes 10
```

### Evaluating Dueling DQN
```bash
python main.py --algorithm dueling_dqn --eval --eval_episodes 10
```

### Evaluating SARSA
```bash
python main.py --algorithm sarsa --eval --eval_episodes 10
```

### Evaluating PPO
```bash
python main.py --algorithm ppo --eval --eval_episodes 10
```

## Algorithm Comparison

### Comparing All Algorithms
```bash
python main.py --compare --algorithms dqn ddqn dueling_dqn sarsa ppo --episodes 1000
```

### Comparing Only Value-Based Methods
```bash
python main.py --compare --algorithms dqn ddqn dueling_dqn sarsa --episodes 1000
```

### Shorter Comparison Run (for testing)
```bash
python main.py --compare --algorithms dqn ddqn dueling_dqn sarsa ppo --episodes 100
```

## Additional Options

### Training with Different Random Seed
```bash
python main.py --algorithm dqn --train --carla_host localhost --carla_port 2000 --episodes 1000 --seed 42
```

### Training with Different Environment Parameters
```bash
python main.py --algorithm dqn --train --carla_host localhost --carla_port 2000 --episodes 1000 --weather WetNoon
```

### Evaluating with a Specific Checkpoint
```bash
python main.py --algorithm dqn --eval --eval_episodes 10 --load_checkpoint models/dqn_20230515-123456/model_final.pth
```

## Statistical Comparison (Multiple Seeds)

For statistical significance, run each algorithm with different seeds:

```bash
for seed in 42 123 456 789 1024; do
    python main.py --algorithm dqn --train --carla_host localhost --carla_port 2000 --episodes 1000 --seed $seed
done
```

## Running Multiple Episodes with Increasing Difficulty

```bash
for weather in ClearNoon CloudyNoon WetNoon HardRainNoon; do
    python main.py --algorithm dqn --eval --eval_episodes 5 --weather $weather
done
```

## Recommended Process for Complete Analysis

1. **Initial Testing**:
   ```bash
   python main.py --algorithm dqn --train --carla_host localhost --carla_port 2000 --episodes 10
   ```

2. **Full Training Comparison**:
   ```bash
   python main.py --compare --algorithms dqn ddqn dueling_dqn sarsa ppo --episodes 1000
   ```

3. **Evaluation of Best Models**:
   ```bash
   for algo in dqn ddqn dueling_dqn sarsa ppo; do
       python main.py --algorithm $algo --eval --eval_episodes 20
   done
   ```

4. **Generate Final Comparison Plots**:
   ```bash
   python main.py --plot-comparison --algorithms dqn ddqn dueling_dqn sarsa ppo
   ``` 