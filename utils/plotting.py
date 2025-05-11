"""
Plotting utilities for comparing RL algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from typing import Dict, List, Tuple
import seaborn as sns

def smooth_curve(points, factor=0.8):
    """
    Apply exponential smoothing to a list of values.
    
    Args:
        points (list): Original data points
        factor (float): Smoothing factor between 0 and 1
            (0: no smoothing, 1: maximum smoothing)
    
    Returns:
        np.array: Smoothed data points
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

def plot_rewards(experiment_results, save_path=None, window=10, figsize=(12, 8)):
    """
    Plot rewards for different algorithms.
    
    Args:
        experiment_results (dict): Dictionary mapping algorithm names to lists of rewards
        save_path (str, optional): Path to save the plot
        window (int): Window size for moving average smoothing
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=figsize)
    
    for algorithm, rewards in experiment_results.items():
        # Convert to numpy array if not already
        rewards_array = np.array(rewards)
        
        # Apply rolling mean for smoothing
        if len(rewards_array) > window:
            smoothed_rewards = pd.Series(rewards_array).rolling(window=window, min_periods=1).mean().values
        else:
            smoothed_rewards = rewards_array
            
        # Plot rewards
        plt.plot(smoothed_rewards, label=algorithm)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_losses(experiment_results, save_path=None, window=10, figsize=(12, 8)):
    """
    Plot training losses for different algorithms.
    
    Args:
        experiment_results (dict): Dictionary mapping algorithm names to lists of losses
        save_path (str, optional): Path to save the plot
        window (int): Window size for moving average smoothing
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=figsize)
    
    for algorithm, losses in experiment_results.items():
        # Convert to numpy array if not already
        losses_array = np.array(losses)
        
        # Apply rolling mean for smoothing
        if len(losses_array) > window:
            smoothed_losses = pd.Series(losses_array).rolling(window=window, min_periods=1).mean().values
        else:
            smoothed_losses = losses_array
            
        # Plot losses
        plt.plot(smoothed_losses, label=algorithm)
    
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss per Episode')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_centerline_deviations(experiment_results, save_path=None, window=10, figsize=(12, 8)):
    """
    Plot centerline deviations for different algorithms.
    
    Args:
        experiment_results (dict): Dictionary mapping algorithm names to lists of centerline deviations
        save_path (str, optional): Path to save the plot
        window (int): Window size for moving average smoothing
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=figsize)
    
    for algorithm, deviations in experiment_results.items():
        # Convert to numpy array if not already
        deviations_array = np.array(deviations)
        
        # Apply rolling mean for smoothing
        if len(deviations_array) > window:
            smoothed_deviations = pd.Series(deviations_array).rolling(window=window, min_periods=1).mean().values
        else:
            smoothed_deviations = deviations_array
            
        # Plot deviations
        plt.plot(smoothed_deviations, label=algorithm)
    
    plt.xlabel('Episode')
    plt.ylabel('Deviation (meters)')
    plt.title('Centerline Deviation per Episode')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_junction_actions(experiment_results, save_path=None, figsize=(14, 10)):
    """
    Plot junction actions distribution for different algorithms.
    
    Args:
        experiment_results (dict): Dictionary mapping algorithm names to dictionaries of action counts
                                 {'left': count, 'right': count, 'forward': count, 'other': count}
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=figsize)
    
    # Prepare data for bar plot
    algorithms = list(experiment_results.keys())
    action_types = ['left', 'right', 'forward', 'other']
    
    # Create matrices for each action type
    left_counts = [experiment_results[alg]['left'] for alg in algorithms]
    right_counts = [experiment_results[alg]['right'] for alg in algorithms]
    forward_counts = [experiment_results[alg]['forward'] for alg in algorithms]
    other_counts = [experiment_results[alg]['other'] for alg in algorithms]
    
    # Set width of bars
    bar_width = 0.2
    
    # Set positions of bars on X axis
    r1 = np.arange(len(algorithms))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Create bars
    plt.bar(r1, left_counts, width=bar_width, label='Left', color='#3274A1')
    plt.bar(r2, right_counts, width=bar_width, label='Right', color='#E1812C')
    plt.bar(r3, forward_counts, width=bar_width, label='Forward', color='#3A923A')
    plt.bar(r4, other_counts, width=bar_width, label='Other', color='#C03D3E')
    
    # Add labels and legend
    plt.xlabel('Algorithm')
    plt.ylabel('Count')
    plt.title('Junction Action Distribution by Algorithm')
    plt.xticks([r + bar_width*1.5 for r in range(len(algorithms))], algorithms)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_all_metrics(algorithm_results, save_path, window=10):
    """
    Plot and compare all metrics for different algorithms.
    
    Args:
        algorithm_results (dict): Dictionary mapping algorithm names to evaluation results
        save_path (str): Path to save the combined plot
        window (int): Window size for moving average smoothing
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot rewards
    for algorithm, results in algorithm_results.items():
        if 'rewards' in results:
            rewards = results['rewards']
            # Apply smoothing if enough data points
            if len(rewards) > window:
                smoothed_rewards = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
            else:
                smoothed_rewards = rewards
                
            # Plot on first subplot
            axs[0, 0].plot(smoothed_rewards, label=algorithm)
    
    axs[0, 0].set_title('Average Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot centerline deviations
    for algorithm, results in algorithm_results.items():
        if 'centerline_deviations' in results and results['centerline_deviations']:
            deviations = results['centerline_deviations']
            # Apply smoothing if enough data points
            if len(deviations) > window:
                smoothed_deviations = pd.Series(deviations).rolling(window=window, min_periods=1).mean().values
            else:
                smoothed_deviations = deviations
                
            # Plot on second subplot
            axs[0, 1].plot(smoothed_deviations, label=algorithm)
    
    axs[0, 1].set_title('Centerline Deviation')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Deviation (meters)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot episode lengths (steps)
    for algorithm, results in algorithm_results.items():
        if 'steps' in results:
            steps = results['steps']
            # Apply smoothing if enough data points
            if len(steps) > window:
                smoothed_steps = pd.Series(steps).rolling(window=window, min_periods=1).mean().values
            else:
                smoothed_steps = steps
                
            # Plot on third subplot
            axs[1, 0].plot(smoothed_steps, label=algorithm)
    
    axs[1, 0].set_title('Episode Length')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Steps')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot junction actions (bar chart)
    algorithms = list(algorithm_results.keys())
    action_types = ['left', 'right', 'forward', 'other']
    
    # Create data arrays for the bar chart
    action_data = {action_type: [] for action_type in action_types}
    
    # Collect data for each algorithm
    for algorithm in algorithms:
        results = algorithm_results[algorithm]
        if 'junction_actions' in results:
            junction_actions = results['junction_actions']
            for action_type in action_types:
                action_data[action_type].append(junction_actions.get(action_type, 0))
        else:
            # Fill with zeros if no junction action data
            for action_type in action_types:
                action_data[action_type].append(0)
    
    # Create bar positions
    x = np.arange(len(algorithms))
    width = 0.2  # Width of bars
    
    # Plot bars for each action type
    for i, action_type in enumerate(action_types):
        axs[1, 1].bar(x + (i - 1.5) * width, action_data[action_type], width, label=action_type)
    
    axs[1, 1].set_title('Junction Actions Distribution')
    axs[1, 1].set_xlabel('Algorithm')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(algorithms)
    axs[1, 1].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python plotting.py <data_dir> <output_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    plot_all_metrics(data_dir, output_dir) 