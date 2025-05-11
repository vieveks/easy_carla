# Reinforcement Learning Analysis and Visualization Guide

This guide explains how to use the analysis and visualization tools for comparing different reinforcement learning algorithms in the CARLA environment.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Available Tools](#available-tools)
- [Understanding Metrics Files](#understanding-metrics-files)
- [Managing Multiple Training Runs](#managing-multiple-training-runs)
- [Continuing Training from Checkpoints](#continuing-training-from-checkpoints)
- [Customizing Visualizations](#customizing-visualizations)
- [Example Workflows](#example-workflows)
- [Common Issues and Solutions](#common-issues-and-solutions)

## Basic Usage

### Step 1: Train Your Agents

First, train your agents using the main script:

```bash
# Train DQN for 1000 episodes
python main.py --algorithm dqn --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000 

# Train DDQN for 1000 episodes
python main.py --algorithm ddqn --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000

# Train Dueling DQN for 1000 episodes
python main.py --algorithm dueling_dqn --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000

# Train SARSA for 1000 episodes
python main.py --algorithm sarsa --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000

# Train PPO for 1000 episodes
python main.py --algorithm ppo --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000
```

During training, metrics are automatically collected and saved to JSON files.

### Step 2: Generate Analysis

Once you have trained multiple algorithms, generate comprehensive analysis and visualizations:

```bash
# Compare all algorithms
python generate_analysis.py --results_dir results --output_dir analysis

# Or compare specific algorithms only
python generate_analysis.py --algorithms dqn ddqn sarsa --output_dir analysis
```

This will create a set of visualizations and a summary report in the specified output directory.

## Available Tools

### 1. `generate_analysis.py`

The main comprehensive analysis tool that generates all visualizations and a detailed report:

```bash
python generate_analysis.py [--results_dir RESULTS_DIR] [--output_dir OUTPUT_DIR] [--algorithms ALGO1 ALGO2 ...]
```

**Arguments:**
- `--results_dir`: Directory containing training results (default: "results")
- `--output_dir`: Directory to save analysis output (default: "analysis")
- `--algorithms`: Specific algorithms to analyze (default: all available)

### 2. `compare_algorithms.py`

A simpler comparison tool that compares algorithms based on final performance:

```bash
python compare_algorithms.py [--algorithms ALGO1 ALGO2 ...] [--results_dir RESULTS_DIR] [--output_dir OUTPUT_DIR]
```

**Arguments:**
- `--algorithms`: List of algorithms to compare (default: all implemented algorithms)
- `--results_dir`: Directory containing algorithm results (default: "results")
- `--output_dir`: Directory to save comparison plots (default: "comparison_plots")

### 3. Direct Visualization Tool

For more custom visualizations:

```bash
python utils/visualization.py --mode [compare|training] [--algorithms ALGO1 ALGO2 ...] [--run_dirs DIR1 DIR2 ...] [--metrics_paths PATH1 PATH2 ...] [--output_dir OUTPUT_DIR]
```

**Modes:**
- `compare`: Compare different algorithms
- `training`: Compare different runs of the same algorithm

## Understanding Metrics Files

During training, the system collects and saves several types of metrics:

1. **Basic Performance Metrics**:
   - Episode rewards, lengths, and losses
   - Evaluation rewards

2. **Safety and Navigation Metrics**:
   - Centerline deviations
   - Collision counts
   - Lane invasion counts
   - Junction actions (left/right turns, forward)

3. **Extended Metrics**:
   - Action distribution
   - Position heatmaps
   - Speed profiles
   - Reward components
   - Computational efficiency (training time, memory usage)

These metrics are saved in JSON files named `{algorithm}_metrics.json` in the training results directory.

## Managing Multiple Training Runs

When you train the same algorithm multiple times, each run creates a separate timestamped directory:

```
results/
├── dqn_20240510-120000/
│   ├── dqn_metrics.json
│   └── ...
├── dqn_20240510-160000/
│   ├── dqn_metrics.json
│   └── ...
└── ...
```

### Handling Multiple Runs

By default, analysis tools select the **most recent** metrics file for each algorithm based on file modification time. This ensures you're comparing the latest version of each algorithm.

### Comparing Specific Runs

To analyze specific runs instead of the latest ones:

1. Specify an exact path to a specific metrics file or directory:
   ```bash
   python generate_analysis.py --results_dir results/dqn_20240510-120000
   ```

2. Manually create a directory with the specific metrics files you want to compare:
   ```bash
   # Copy specific metrics files
   mkdir my_comparison
   cp results/dqn_20240510-120000/dqn_metrics.json my_comparison/
   cp results/ddqn_20240510-130000/ddqn_metrics.json my_comparison/
   
   # Run analysis on these specific files
   python generate_analysis.py --results_dir my_comparison
   ```

### Comparing Multiple Runs of the Same Algorithm

To compare different runs of the same algorithm (e.g., different hyperparameters or random seeds):

```bash
python utils/visualization.py --mode training --metrics_paths path/to/run1/metrics.json path/to/run2/metrics.json --algorithms "Run 1" "Run 2"
```

This visualizes multiple runs so you can see the effect of different seeds or parameters on the same algorithm.

## Continuing Training from Checkpoints

You can restart training from a previously saved checkpoint, which is useful if:
- Training was interrupted and you want to continue
- You want to extend training for more episodes
- You're fine-tuning a model with different parameters

### Method 1: Explicit Checkpoint Loading

Specify a checkpoint file directly using the `--load_checkpoint` argument:

```bash
# Find the latest checkpoint for DQN
ls -t results/dqn_*/*.pth | head -1
# Example output: results/dqn_20240510-120000/model_final.pth

# Continue training from that checkpoint
python main.py --algorithm dqn --train --episodes 500 --carla_host 127.0.0.1 --carla_port 2000 --load_checkpoint results/dqn_20240510-120000/model_final.pth
```

### Method 2: Automatic Checkpoint Detection

When no checkpoint is specified, the script will automatically try to find and load the latest checkpoint:

```bash
# This will automatically find and load the latest checkpoint for DQN
python main.py --algorithm dqn --train --episodes 500 --carla_host 127.0.0.1 --carla_port 2000
```

### Quick Reference for Continued Training

Here are the commands to continue training for each algorithm:

```bash
# Continue DQN training
python main.py --algorithm dqn --train --episodes 500 --carla_host 127.0.0.1 --carla_port 2000 --load_checkpoint results/dqn_*/model_final.pth

# Continue DDQN training
python main.py --algorithm ddqn --train --episodes 500 --carla_host 127.0.0.1 --carla_port 2000 --load_checkpoint results/ddqn_*/model_final.pth

# Continue SARSA training
python main.py --algorithm sarsa --train --episodes 500 --carla_host 127.0.0.1 --carla_port 2000 --load_checkpoint results/sarsa_*/model_final.pth

# Continue Dueling DQN training
python main.py --algorithm dueling_dqn --train --episodes 500 --carla_host 127.0.0.1 --carla_port 2000 --load_checkpoint results/dueling_dqn_*/model_final.pth

# Continue PPO training
python main.py --algorithm ppo --train --episodes 500 --carla_host 127.0.0.1 --carla_port 2000 --load_checkpoint results/ppo_*/model_final.pth
```

### Important Considerations

1. **New results directory**: Continuing training creates a new timestamped directory for checkpoints and metrics.
   
2. **Metrics continuity**: Model weights and training state will be preserved, but metrics might show a discontinuity.

3. **Hyperparameter consistency**: Ensure you use the same hyperparameters as the original training run.

4. **Environment changes**: If the environment has changed significantly, it might be better to start fresh.

### Technical Implementation Details

The checkpoint system is implemented consistently across all algorithms. Here's a technical overview of how it works:

#### Checkpoint Contents

Each checkpoint file (.pth) contains:

1. **Neural Network Model State**: Weights, biases, and other parameters of the model
2. **Optimizer State**: Learning rates, momentum values, and optimizer-specific parameters
3. **Training Progress Information**:
   - Current training step
   - Epsilon values (for exploration in DQN-family algorithms)
   - Algorithm-specific hyperparameters

For example, in DQN:
```python
# From dqn.py
additional_info = {
    'training_step': self.training_step,
    'epsilon_start': self.epsilon_start,
    'epsilon_end': self.epsilon_end,
    'epsilon_decay': self.epsilon_decay
}
save_model(self.q_network, self.optimizer, file_path, additional_info)
```

#### Saving Metrics

Alongside model checkpoints, each algorithm saves:
- Performance metrics (rewards, losses)
- Extended metrics (centerline deviations, collision counts, etc.)
- Algorithm-specific metrics (e.g., junction actions for PPO)

These are stored in JSON files in the same directory as the checkpoint.

#### Loading Process

When loading a checkpoint, the system:
1. Restores model parameters
2. Restores optimizer state
3. Updates training progress variables
4. Loads saved metrics

```python
# Example from dqn.py
self.q_network, self.optimizer, additional_info = load_model(
    self.q_network,
    self.optimizer,
    file_path
)
self.training_step = additional_info.get('training_step', 0)
```

#### Target Networks

For algorithms with target networks (DQN, DDQN, Dueling DQN), the target network state is also saved and restored:

```python
# Save target network
target_path = os.path.join(os.path.dirname(file_path), 'dqn_target_model.pth')
torch.save(self.target_network.state_dict(), target_path)

# Load target network
target_path = os.path.join(os.path.dirname(file_path), 'dqn_target_model.pth')
if os.path.exists(target_path):
    self.target_network.load_state_dict(torch.load(target_path))
```

#### Resume Training Flow

When resuming training, the system:
1. Initializes a new agent with the specified algorithm
2. Loads the checkpoint (if specified or auto-detected)
3. Creates a new results directory for continued training
4. Continues training from the saved training step
5. Preserves all training state including replay buffers and exploration parameters

#### Results Directory Handling

When continuing training from a checkpoint, the system **always creates a new timestamped results directory**. For example:

```
Original training:
results/dqn_20240510-120000/  (contains original metrics and model)

After continuing training:
results/dqn_20240510-120000/  (contains original metrics and model)
results/dqn_20240511-143000/  (contains continued training metrics and final model)
```

The new directory will contain:
- New metrics files that start from where the checkpoint left off
- Updated model checkpoints reflecting continued training
- New evaluation results and visualizations

This approach provides several benefits:
1. Original results are preserved intact for comparison
2. You can easily identify different training runs/stages
3. If continued training performs worse, you can revert to the original checkpoint
4. Analysis tools can compare both original and continued training results

When running analysis with `generate_analysis.py`, it will by default use the latest metrics file for each algorithm. If you want to analyze the original training run instead, you'll need to specify the exact directory:

```bash
# Analyze the original training run specifically
python generate_analysis.py --results_dir results/dqn_20240510-120000
```

## Customizing Visualizations

The visualization tools generate several types of plots:

1. **Reward Curves**: Episode rewards during training
2. **Learning Efficiency**: Reward vs. environment steps
3. **Loss Curves**: Training loss over time
4. **Centerline Deviation**: Lane following accuracy
5. **Collision Rates**: Safety performance
6. **Action Distribution**: Driving behavior analysis
7. **Position Heatmaps**: Spatial coverage
8. **Final Performance Comparison**: Summary of key metrics
9. **Computational Efficiency**: Training time and resource usage

To customize these visualizations, you can directly modify the relevant functions in `utils/visualization.py`.

## Example Workflows

### Basic Comparison Workflow

```bash
# Train multiple algorithms
python main.py --algorithm dqn --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000
python main.py --algorithm ddqn --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000
python main.py --algorithm sarsa --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000

# Generate comprehensive analysis
python generate_analysis.py

# View the results in the "analysis" directory
```

### Hyperparameter Tuning Workflow

```bash
# Train multiple versions of the same algorithm
python main.py --algorithm dqn --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000
# Modify hyperparameters in config.py
python main.py --algorithm dqn --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000

# Compare the different runs
mkdir dqn_comparison
cp results/dqn_*/*.json dqn_comparison/
python utils/visualization.py --mode training --metrics_paths dqn_comparison/*.json --algorithms "Default" "Modified" --output_dir dqn_comparison_results
```

### Ablation Study Workflow

```bash
# For each variant in your ablation study:
# 1. Modify the algorithm implementation
# 2. Train the modified version
python main.py --algorithm modified_dqn --train --episodes 1000 --carla_host 127.0.0.1 --carla_port 2000

# Collect metrics from each variant
mkdir ablation_study
cp results/*/*.json ablation_study/

# Generate comparison
python generate_analysis.py --results_dir ablation_study --output_dir ablation_results
```

## Common Issues and Solutions

### Visualization Warnings

When running the analysis scripts, you might see warnings like:

```
No artists with labels found to put in legend.
FixedFormatter should only be used together with FixedLocator
Mean of empty slice.
invalid value encountered in double_scalars
posx and posy should be finite values
```

These warnings are normal and occur when:
- Some metrics weren't collected or have missing values
- Training ended early or was interrupted
- Certain metrics are empty for specific algorithms

The analysis will still generate useful visualizations for the available data. 