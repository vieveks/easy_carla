#!/usr/bin/env python
"""
Main script for training and evaluating reinforcement learning algorithms
on the CARLA environment.
"""
import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("carla_rl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import configurations and utilities
import config
from carla_env.rl_env import CarlaEnv
from utils.helpers import set_random_seed, ensure_directory, preprocess_image
from utils.plotting import plot_all_metrics

# Import algorithms
from algorithms.dqn import DQNAgent
from algorithms.ddqn import DDQNAgent
from algorithms.dueling_dqn import DuelingDQNAgent
from algorithms.ppo import PPOAgent
from algorithms.sarsa import SARSAAgent

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CARLA Reinforcement Learning")
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'ddqn', 'dueling_dqn', 'ppo', 'sarsa'],
                        help='RL algorithm to use')
    
    # Mode selection
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate the agent')
    parser.add_argument('--compare', action='store_true', help='Compare all algorithms')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save/load models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Environment parameters
    parser.add_argument('--carla_host', type=str, default=config.CARLA_HOST,
                        help='CARLA server host')
    parser.add_argument('--carla_port', type=int, default=config.CARLA_PORT,
                        help='CARLA server port')
    parser.add_argument('--carla_map', type=str, default=config.CARLA_MAP,
                        help='CARLA map to use')
    parser.add_argument('--image_height', type=int, default=84,
                        help='Height of observation images')
    parser.add_argument('--image_width', type=int, default=84,
                        help='Width of observation images')
    
    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation episodes')
    
    # Checkpoint parameters
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='Save model every N episodes')
    
    return parser.parse_args()

def create_agent(args):
    """
    Create a reinforcement learning agent based on the specified algorithm.
    
    Args:
        args: Command line arguments
        
    Returns:
        The created agent
    """
    # Determine state shape
    state_shape = (3, args.image_height, args.image_width)  # RGB image
    
    # Number of possible actions
    action_size = len(config.DISCRETE_ACTIONS)
    
    # Common parameters for all agents
    common_params = {
        'state_shape': state_shape,
        'action_size': action_size,
        'device': None,  # Auto-select device (CPU or CUDA)
        'model_dir': args.model_dir,
        'tensorboard_dir': 'logs'
    }
    
    # Training parameters that aren't passed to agent constructor
    training_params = ['training_episodes', 'save_interval', 'eval_interval', 'log_dir', 'results_dir']
    
    # Create agent based on algorithm
    if args.algorithm == 'dqn':
        # Get parameters from config
        all_params = config.DQN_HYPERPARAMS.copy()
        
        # Extract only parameters expected by DQNAgent constructor
        agent_params = {k: v for k, v in all_params.items() if k not in training_params}
        agent_params.update(common_params)
        
        # Create agent
        agent = DQNAgent(**agent_params)
        
        # Store hyperparameters for training
        agent.hyperparams = all_params
        
        return agent
    
    elif args.algorithm == 'ddqn':
        # Get parameters from config
        all_params = config.DDQN_HYPERPARAMS.copy()
        
        # Extract only parameters expected by DDQNAgent constructor
        agent_params = {k: v for k, v in all_params.items() if k not in training_params}
        agent_params.update(common_params)
        
        # Create agent
        agent = DDQNAgent(**agent_params)
        
        # Store hyperparameters for training
        agent.hyperparams = all_params
        
        return agent
    
    elif args.algorithm == 'dueling_dqn':
        # Get parameters from config
        all_params = config.DUELING_DQN_HYPERPARAMS.copy()
        
        # Extract only parameters expected by DuelingDQNAgent constructor
        agent_params = {k: v for k, v in all_params.items() if k not in training_params}
        agent_params.update(common_params)
        
        # Create agent
        agent = DuelingDQNAgent(**agent_params)
        
        # Store hyperparameters for training
        agent.hyperparams = all_params
        
        return agent
    
    elif args.algorithm == 'ppo':
        # Get parameters from config
        all_params = config.PPO_HYPERPARAMS.copy()
        
        # Extract only parameters expected by PPOAgent constructor
        agent_params = {k: v for k, v in all_params.items() if k not in training_params}
        agent_params.update(common_params)
        
        # Create agent
        agent = PPOAgent(**agent_params)
        
        # Store hyperparameters for training
        agent.hyperparams = all_params
        
        return agent
    
    elif args.algorithm == 'sarsa':
        # Get parameters from config
        all_params = config.SARSA_HYPERPARAMS.copy()
        
        # Extract only parameters expected by SARSAAgent constructor
        agent_params = {k: v for k, v in all_params.items() if k not in training_params}
        agent_params.update(common_params)
        
        # Create agent
        agent = SARSAAgent(**agent_params)
        
        # Store hyperparameters for training
        agent.hyperparams = all_params
        
        return agent
    
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

def create_environment(args):
    """
    Create the CARLA environment.
    
    Args:
        args: Command line arguments
        
    Returns:
        The created environment
    """
    # Create environment
    env = CarlaEnv(
        host=args.carla_host,
        port=args.carla_port,
        town=args.carla_map,
        image_shape=(args.image_height, args.image_width, 3)
    )
    
    return env

def train(args, agent, env):
    """
    Train the agent on the environment.
    
    Args:
        args: Command line arguments
        agent: The RL agent
        env: The environment
    """
    # Set up directories
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.model_dir, f"{args.algorithm}_{timestamp}")
    ensure_directory(run_dir)
    
    # Set up logging to file
    log_file = os.path.join(run_dir, "training_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Save configuration
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        config_dict = vars(args)
        config_dict.update(agent.hyperparams)
        json.dump(config_dict, f, indent=4, default=str)
    
    # Create results directory for this run
    run_results_dir = os.path.join(args.results_dir, f"{args.algorithm}_{timestamp}")
    ensure_directory(run_results_dir)
    
    # Log run directory information
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Results directory: {run_results_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Determine number of episodes
    num_episodes = args.episodes if args.episodes is not None else agent.hyperparams['training_episodes']
    
    # Determine save interval
    save_interval = args.save_interval if args.save_interval is not None else agent.hyperparams.get('save_interval', 50)
    
    # Train the agent
    logger.info(f"Training {args.algorithm} for {num_episodes} episodes...")
    
    # Initialize progress bar
    progress_bar = tqdm(total=num_episodes, desc=f"Training {args.algorithm}")
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        agent.load(args.load_checkpoint)
    
    # Initialize metrics for logging
    episode_metrics = []
    
    # Training loop
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        
        # Initialize episode statistics
        done = False
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Apply action to environment
            next_state, reward, done, info = env.step(action)
            
            # Clip reward to prevent extreme values
            clipped_reward = np.clip(reward, -10.0, 10.0)
            
            # Store experience in agent's memory
            agent.store_experience(state, action, clipped_reward, next_state, done)
            
            # Update extended metrics
            agent.update_extended_metrics(state, action, reward, info)
            
            # Track junction actions if at a junction
            if info and info.get('is_junction', False):
                agent.log_junction_action(action)
                
            # Track centerline deviation
            if info and 'centerline_offset' in info:
                agent.log_centerline_deviation(info['centerline_offset'])
            
            # Train agent
            loss = agent.train_step()
            
            # Update episode statistics
            episode_reward += reward  # Keep original reward for logging
            if loss is not None:
                episode_loss += loss
            
            # Update state
            state = next_state
            step_count += 1
            
            # Check if episode has reached max steps
            if step_count >= env.max_episode_steps:
                break
        
        # Calculate average loss for the episode
        avg_loss = episode_loss / step_count if step_count > 0 else 0
        
        # Save performance metrics
        agent.save_metrics(episode, episode_reward, avg_loss)
        
        # Get episode statistics
        episode_stats = env.get_episode_statistics()
        agent.save_additional_metrics(episode, episode_stats)
        
        # Record metrics for logging
        metric_entry = {
            'episode': episode,
            'reward': episode_reward,
            'loss': avg_loss,
            'steps': step_count,
            'centerline_deviation': episode_stats.get('avg_centerline_deviation', 0),
            'collision_count': episode_stats.get('collision_count', 0),
            'lane_invasion_count': episode_stats.get('lane_invasion_count', 0)
        }
        episode_metrics.append(metric_entry)
        
        # Log episode results
        logger.info(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Loss = {avg_loss:.4f}, Steps = {step_count}")
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'loss': f'{avg_loss:.4f}'
        })
        
        # Save checkpoint and logs
        if save_interval > 0 and (episode + 1) % save_interval == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(run_dir, f"checkpoint_episode_{episode+1}.pth")
            agent.save(checkpoint_path)
            
            # Save metrics to CSV
            metrics_df_path = os.path.join(run_dir, "episode_metrics.csv")
            try:
                import pandas as pd
                pd.DataFrame(episode_metrics).to_csv(metrics_df_path, index=False)
            except ImportError:
                with open(metrics_df_path, 'w') as f:
                    f.write(','.join(episode_metrics[0].keys()) + '\n')
                    for m in episode_metrics:
                        f.write(','.join(str(m[k]) for k in episode_metrics[0].keys()) + '\n')
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            logger.info(f"Saved metrics to {metrics_df_path}")
    
    # Save final model
    final_model_path = os.path.join(run_dir, "model_final.pth")
    agent.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Close progress bar
    progress_bar.close()
    
    # Plot learning curves
    plot_path = os.path.join(run_results_dir, "learning_curves.png")
    agent.plot_learning_curves(plot_path)
    logger.info(f"Saved learning curves to {plot_path}")
    
    # Save results to file
    results_path = os.path.join(run_results_dir, "results.json")
    metrics_json_path = os.path.join(run_results_dir, f"{args.algorithm}_metrics.json")
    agent.save_all_metrics(metrics_json_path)
    logger.info(f"Saved extended metrics to {metrics_json_path}")
    
    # Remove file handler to avoid duplicate log entries
    logger.removeHandler(file_handler)
    
    return agent

def evaluate(args, agent, env):
    """
    Evaluate the agent on the environment.
    
    Args:
        args: Command line arguments
        agent: The RL agent
        env: The environment
    """
    # Create a timestamped directory for evaluation results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    eval_dir = os.path.join(args.results_dir, f"{args.algorithm}_eval_{timestamp}")
    ensure_directory(eval_dir)
    
    # Set up logging to file
    log_file = os.path.join(eval_dir, "evaluation_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Load model if path specified
    if args.load_checkpoint:
        logger.info(f"Loading model from {args.load_checkpoint}")
        agent.load(args.load_checkpoint)
    else:
        # Try to find the most recent run directory and load the final model
        run_dirs = [d for d in os.listdir(args.model_dir) if d.startswith(f"{args.algorithm}_") and os.path.isdir(os.path.join(args.model_dir, d))]
        if run_dirs:
            # Sort by timestamp (most recent first)
            run_dirs.sort(reverse=True)
            latest_run_dir = os.path.join(args.model_dir, run_dirs[0])
            final_model_path = os.path.join(latest_run_dir, "model_final.pth")
            
            if os.path.exists(final_model_path):
                logger.info(f"Loading final model from most recent run: {final_model_path}")
                agent.load(final_model_path)
            else:
                # Try to find the latest checkpoint in this directory
                checkpoints = [f for f in os.listdir(latest_run_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
                if checkpoints:
                    # Sort by episode number (highest first)
                    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
                    latest_checkpoint = os.path.join(latest_run_dir, checkpoints[0])
                    logger.info(f"Loading latest checkpoint: {latest_checkpoint}")
                    agent.load(latest_checkpoint)
                else:
                    logger.warning(f"No models found in the latest run directory: {latest_run_dir}. Using untrained model.")
        else:
            logger.warning(f"No runs found in model directory. Using untrained model.")
    
    # Switch agent to evaluation mode
    agent.eval()
    
    # Run evaluation episodes
    logger.info(f"Evaluating {args.algorithm} for {args.eval_episodes} episodes...")
    
    eval_results = {
        'rewards': [],
        'steps': [],
        'centerline_deviations': [],
        'junction_actions': {
            'left': 0,
            'right': 0,
            'forward': 0,
            'other': 0
        },
        'collision_count': 0,
        'lane_invasion_count': 0
    }
    
    # For detailed logging
    episode_metrics = []
    
    for episode in range(args.eval_episodes):
        # Reset environment
        state = env.reset()
        
        # Initialize episode statistics
        done = False
        episode_reward = 0
        step_count = 0
        
        # Episode loop
        while not done:
            # Select action (without exploration)
            action = agent.select_action(state, eval_mode=True)
            
            # Apply action to environment
            next_state, reward, done, info = env.step(action)
            
            # Update extended metrics
            agent.update_extended_metrics(state, action, reward, info)
            
            # Track junction actions if at a junction
            if info and info.get('is_junction', False):
                agent.log_junction_action(action)
                
            # Track centerline deviation
            if info and 'centerline_offset' in info:
                agent.log_centerline_deviation(info['centerline_offset'])
            
            # Update episode statistics
            episode_reward += reward
            step_count += 1
            
            # Update state
            state = next_state
            
            # Check if episode has reached max steps
            if step_count >= env.max_episode_steps:
                break
        
        # Get episode statistics
        episode_stats = env.get_episode_statistics()
        
        # Update evaluation results
        eval_results['rewards'].append(episode_reward)
        eval_results['steps'].append(step_count)
        
        # Update collision and lane invasion counts
        eval_results['collision_count'] += episode_stats.get('collision_count', 0)
        eval_results['lane_invasion_count'] += episode_stats.get('lane_invasion_count', 0)
        
        if 'avg_centerline_deviation' in episode_stats:
            eval_results['centerline_deviations'].append(episode_stats['avg_centerline_deviation'])
        
        # Aggregate junction actions
        if 'junction_actions' in episode_stats:
            for action_type, count in episode_stats['junction_actions'].items():
                if action_type in eval_results['junction_actions']:
                    eval_results['junction_actions'][action_type] += count
        
        # Record detailed metrics for this episode
        metric_entry = {
            'episode': episode,
            'reward': episode_reward,
            'steps': step_count,
            'centerline_deviation': episode_stats.get('avg_centerline_deviation', 0),
            'collision_count': episode_stats.get('collision_count', 0),
            'lane_invasion_count': episode_stats.get('lane_invasion_count', 0)
        }
        episode_metrics.append(metric_entry)
        
        logger.info(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    # Calculate average metrics
    avg_reward = np.mean(eval_results['rewards'])
    avg_steps = np.mean(eval_results['steps'])
    avg_centerline = np.mean(eval_results['centerline_deviations']) if len(eval_results['centerline_deviations']) > 0 else 0
    
    logger.info(f"Evaluation results for {args.algorithm}:")
    logger.info(f"  Average reward: {avg_reward:.2f}")
    logger.info(f"  Average episode length: {avg_steps:.2f}")
    logger.info(f"  Average centerline deviation: {avg_centerline:.2f}")
    logger.info(f"  Total collisions: {eval_results['collision_count']}")
    logger.info(f"  Total lane invasions: {eval_results['lane_invasion_count']}")
    
    # Save evaluation results
    results_path = os.path.join(eval_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    # Save detailed metrics to CSV
    metrics_df_path = os.path.join(eval_dir, "episode_metrics.csv")
    try:
        import pandas as pd
        pd.DataFrame(episode_metrics).to_csv(metrics_df_path, index=False)
    except ImportError:
        with open(metrics_df_path, 'w') as f:
            f.write(','.join(episode_metrics[0].keys()) + '\n')
            for m in episode_metrics:
                f.write(','.join(str(m[k]) for k in episode_metrics[0].keys()) + '\n')
    
    logger.info(f"Saved evaluation results to {eval_dir}")
    
    # Remove file handler
    logger.removeHandler(file_handler)
    
    return eval_results

def compare_algorithms(args):
    """
    Compare the performance of all implemented algorithms.
    
    Args:
        args: Command line arguments
    """
    # Create timestamp for this comparison
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    compare_dir = os.path.join(args.results_dir, f"comparison_{timestamp}")
    ensure_directory(compare_dir)
    
    # Set up logging
    log_file = os.path.join(compare_dir, "comparison_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # List all implemented algorithms
    algorithms = ['dqn', 'ddqn', 'dueling_dqn', 'ppo', 'sarsa']
    
    # Collect results for each algorithm
    algorithm_results = {}
    
    for algo in algorithms:
        # Find evaluation results for this algorithm
        eval_dirs = [d for d in os.listdir(args.results_dir) if d.startswith(f"{algo}_eval_") and os.path.isdir(os.path.join(args.results_dir, d))]
        
        if eval_dirs:
            # Sort by timestamp (most recent first)
            eval_dirs.sort(reverse=True)
            latest_eval_dir = os.path.join(args.results_dir, eval_dirs[0])
            
            # Check for evaluation results
            eval_results_path = os.path.join(latest_eval_dir, "eval_results.json")
            if os.path.exists(eval_results_path):
                with open(eval_results_path, 'r') as f:
                    eval_results = json.load(f)
                algorithm_results[algo] = eval_results
                logger.info(f"Loaded evaluation results for {algo} from {eval_results_path}")
            else:
                logger.warning(f"No evaluation results found for {algo} in {latest_eval_dir}")
        else:
            # Try looking in the old directory structure
            old_eval_path = os.path.join(args.results_dir, f"{algo}_eval_results.json")
            if os.path.exists(old_eval_path):
                with open(old_eval_path, 'r') as f:
                    eval_results = json.load(f)
                algorithm_results[algo] = eval_results
                logger.info(f"Loaded evaluation results for {algo} from {old_eval_path} (old format)")
            else:
                logger.warning(f"No evaluation results found for {algo}")
    
    # Plot comparison if at least one algorithm has results
    if algorithm_results:
        logger.info("Generating comparison plots...")
        
        # Plot combined metrics
        plot_path = os.path.join(compare_dir, "algorithm_comparison.png")
        plot_all_metrics(algorithm_results, plot_path)
        
        # Save comparison data
        comparison_data_path = os.path.join(compare_dir, "comparison_data.json")
        with open(comparison_data_path, 'w') as f:
            json.dump(algorithm_results, f, indent=4)
        
        logger.info(f"Saved comparison plots to {plot_path}")
        logger.info(f"Saved comparison data to {comparison_data_path}")
    else:
        logger.error("No evaluation results found for any algorithm. Please run evaluation first.")
    
    # Remove file handler
    logger.removeHandler(file_handler)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create directories
    ensure_directory(args.model_dir)
    ensure_directory(args.results_dir)
    
    # Handle comparison mode
    if args.compare:
        compare_algorithms(args)
        return
    
    # Create agent
    agent = create_agent(args)
    
    # Create environment
    env = create_environment(args)
    
    try:
        # Train agent
        if args.train:
            train(args, agent, env)
        
        # Evaluate agent
        if args.eval:
            evaluate(args, agent, env)
        
        # If neither train nor eval specified, train by default
        if not args.train and not args.eval:
            logger.info("No mode specified, defaulting to training.")
            train(args, agent, env)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if env:
            env.close()
        
        logger.info("Done.")

if __name__ == "__main__":
    main() 