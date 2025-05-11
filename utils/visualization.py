"""
Visualization tools for analyzing and comparing reinforcement learning agents
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from collections import defaultdict
import matplotlib.gridspec as gridspec
import cv2
import pygame
import threading
import time
import queue
from collections import deque

# Define consistent colors for algorithms
ALGORITHM_COLORS = {
    'dqn': '#1f77b4',      # blue
    'ddqn': '#ff7f0e',     # orange
    'dueling_dqn': '#2ca02c',  # green
    'sarsa': '#d62728',    # red
    'ppo': '#9467bd',      # purple
}

class CameraViewer:
    """
    Class for displaying camera input from CARLA in a separate window.
    This provides a real-time view of what the agent is seeing.
    """
    def __init__(self, window_name="Camera View", width=400, height=300, max_queue_size=5):
        """
        Initialize the camera viewer.
        
        Args:
            window_name (str): Name of the window
            width (int): Width of the display window
            height (int): Height of the display window
            max_queue_size (int): Maximum size of the frame queue
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        
        # Initialize pygame for display
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(window_name)
        
        # Initialize a clock for controlling FPS
        self.clock = pygame.time.Clock()
        self.fps = 20  # Target FPS
    
    def start(self):
        """Start the viewer thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._display_loop)
            self.thread.daemon = True  # Thread will exit when main program exits
            self.thread.start()
            print(f"Camera viewer started: {self.window_name}")
    
    def stop(self):
        """Stop the viewer thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        pygame.quit()
        print(f"Camera viewer stopped: {self.window_name}")
    
    def update(self, frame):
        """
        Update the display with a new frame.
        
        Args:
            frame: The camera frame to display
        """
        if self.running:
            # Don't block if queue is full (just drop frames)
            try:
                # Ensure frame is the right shape and dtype
                if frame is not None:
                    # Resize the frame if needed
                    if frame.shape[0] != self.height or frame.shape[1] != self.width:
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    # Ensure frame is RGB format
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 4:  # RGBA
                        frame = frame[:, :, :3]  # Drop alpha channel
                    
                    # Ensure the frame is uint8
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full
    
    def _display_loop(self):
        """Main display loop running in a separate thread"""
        last_frame = None
        
        while self.running:
            # Check for pygame events (e.g., window close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
            
            # Get newest frame from queue if available
            try:
                frame = self.frame_queue.get(block=False)
                last_frame = frame
                self.frame_queue.task_done()
            except queue.Empty:
                frame = last_frame  # Keep displaying last frame if no new ones
            
            # Display the frame if we have one
            if frame is not None:
                # Convert frame to pygame surface
                frame = np.flipud(np.fliplr(frame))  # Flip to match pygame's coordinate system
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                
                # Draw to screen
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()
            
            # Control the frame rate
            self.clock.tick(self.fps)

def load_metrics(file_path):
    """
    Load metrics from a JSON file.
    
    Args:
        file_path (str): Path to the metrics file
        
    Returns:
        dict: Loaded metrics
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics from {file_path}: {e}")
        return {}

def smooth_curve(points, factor=0.9):
    """
    Apply exponential smoothing to a curve.
    
    Args:
        points (list): Data points to smooth
        factor (float): Smoothing factor (higher = smoother)
        
    Returns:
        list: Smoothed curve
    """
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

def plot_reward_curves(algorithm_data, save_path=None, smoothing_factor=0.9):
    """
    Plot reward curves for multiple algorithms.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        save_path (str, optional): Path to save the figure
        smoothing_factor (float): Smoothing factor for the curves
    """
    plt.figure(figsize=(12, 8))
    
    for algo, data in algorithm_data.items():
        if 'episode_rewards' in data and len(data['episode_rewards']) > 0:
            rewards = data['episode_rewards']
            episodes = list(range(1, len(rewards) + 1))
            
            # Plot raw data with low alpha
            plt.plot(episodes, rewards, alpha=0.2, color=ALGORITHM_COLORS.get(algo, None))
            
            # Plot smoothed curve
            smoothed_rewards = smooth_curve(rewards, smoothing_factor)
            plt.plot(episodes, smoothed_rewards, label=algo.upper(), linewidth=2, 
                     color=ALGORITHM_COLORS.get(algo, None))
    
    plt.title('Episode Rewards During Training', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reward curves to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_learning_efficiency(algorithm_data, save_path=None):
    """
    Plot learning efficiency (reward vs. steps) for multiple algorithms.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    for algo, data in algorithm_data.items():
        if 'episode_rewards' in data and 'episode_lengths' in data:
            rewards = data['episode_rewards']
            lengths = data['episode_lengths']
            
            # Calculate cumulative steps
            cum_steps = np.cumsum(lengths)
            
            # Plot reward vs. steps
            plt.plot(cum_steps, rewards, label=algo.upper(), linewidth=2,
                     color=ALGORITHM_COLORS.get(algo, None))
    
    plt.title('Learning Efficiency: Reward vs. Environment Steps', fontsize=16)
    plt.xlabel('Environment Steps', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved efficiency plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_loss_curves(algorithm_data, save_path=None):
    """
    Plot loss curves for multiple algorithms.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    for algo, data in algorithm_data.items():
        if 'losses' in data and len(data['losses']) > 0:
            losses = data['losses']
            episodes = list(range(1, len(losses) + 1))
            
            # Plot loss curve
            plt.plot(episodes, losses, label=algo.upper(), linewidth=2,
                     color=ALGORITHM_COLORS.get(algo, None))
    
    plt.title('Training Loss', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.yscale('log')  # Use log scale for better visualization
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_centerline_deviation(algorithm_data, save_path=None):
    """
    Plot centerline deviation for multiple algorithms.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    for algo, data in algorithm_data.items():
        if 'episode_stats' in data and 'avg_centerline_deviation' in data['episode_stats']:
            deviations = data['episode_stats']['avg_centerline_deviation']
            episodes = list(range(1, len(deviations) + 1))
            
            # Plot centerline deviation
            plt.plot(episodes, deviations, label=algo.upper(), linewidth=2,
                     color=ALGORITHM_COLORS.get(algo, None))
    
    plt.title('Average Centerline Deviation', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Deviation (m)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved centerline deviation plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_collision_rates(algorithm_data, save_path=None):
    """
    Plot collision rates for multiple algorithms.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    for algo, data in algorithm_data.items():
        if 'episode_stats' in data and 'collision_count' in data['episode_stats']:
            collisions = data['episode_stats']['collision_count']
            episodes = list(range(1, len(collisions) + 1))
            
            # Calculate moving average of collision rate
            window_size = min(10, len(collisions))
            if window_size > 0:
                collision_rate = []
                for i in range(len(collisions)):
                    start = max(0, i - window_size + 1)
                    rate = sum(collisions[start:i+1]) / (i - start + 1)
                    collision_rate.append(rate)
                
                # Plot collision rate
                plt.plot(episodes, collision_rate, label=algo.upper(), linewidth=2,
                         color=ALGORITHM_COLORS.get(algo, None))
    
    plt.title('Collision Rate (Moving Average)', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Collisions per Episode', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved collision rate plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_action_distribution(algorithm_data, save_path=None):
    """
    Plot action distribution for multiple algorithms.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        save_path (str, optional): Path to save the figure
    """
    # Create a figure with subplots for each algorithm
    n_algos = len(algorithm_data)
    fig, axes = plt.subplots(1, n_algos, figsize=(5*n_algos, 6), sharey=True)
    
    if n_algos == 1:
        axes = [axes]  # Make it iterable for single algorithm case
    
    for i, (algo, data) in enumerate(algorithm_data.items()):
        if 'action_distribution' in data:
            # Extract action counts
            actions = list(data['action_distribution'].keys())
            counts = list(data['action_distribution'].values())
            
            # Define action labels
            action_labels = {
                '0': 'No action',
                '1': 'Throttle',
                '2': 'Left',
                '3': 'Right',
                '4': 'Throttle+Left',
                '5': 'Throttle+Right',
                '6': 'Brake'
            }
            labels = [action_labels.get(a, a) for a in actions]
            
            # Plot bar chart
            bars = axes[i].bar(labels, counts, color=ALGORITHM_COLORS.get(algo, None))
            axes[i].set_title(f'{algo.upper()} Action Distribution', fontsize=14)
            axes[i].set_xlabel('Action', fontsize=12)
            if i == 0:
                axes[i].set_ylabel('Count', fontsize=12)
            
            # Rotate x-axis labels
            axes[i].set_xticklabels(labels, rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved action distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_position_heatmap(algorithm_data, algo_name, save_path=None):
    """
    Plot position heatmap for a specific algorithm.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        algo_name (str): Algorithm to plot
        save_path (str, optional): Path to save the figure
    """
    if algo_name not in algorithm_data:
        print(f"Algorithm {algo_name} not found in data")
        return
    
    data = algorithm_data[algo_name]
    if 'position_heatmap' not in data or not data['position_heatmap']:
        print(f"No position heatmap data available for {algo_name}")
        return
    
    # Extract position data
    heatmap_data = data['position_heatmap']
    
    # Convert string keys to coordinates
    coordinates = []
    counts = []
    for pos_key, count in heatmap_data.items():
        try:
            x, y = map(int, pos_key.split('-'))
            coordinates.append((x, y))
            counts.append(count)
        except:
            continue
    
    if not coordinates:
        print(f"No valid coordinates found in position heatmap for {algo_name}")
        return
    
    # Create a grid for the heatmap
    x_coords = [c[0] for c in coordinates]
    y_coords = [c[1] for c in coordinates]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    grid_size = (x_max - x_min + 1, y_max - y_min + 1)
    grid = np.zeros(grid_size)
    
    # Fill the grid with counts
    for (x, y), count in zip(coordinates, counts):
        grid[x - x_min, y - y_min] = count
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid, cmap='viridis', cbar_kws={'label': 'Visit Count'})
    
    plt.title(f'{algo_name.upper()} Position Heatmap', fontsize=16)
    plt.xlabel('X Coordinate (5m bins)', fontsize=14)
    plt.ylabel('Y Coordinate (5m bins)', fontsize=14)
    
    # Adjust tick labels to show actual coordinates
    x_ticks = np.arange(0, grid_size[0], 5)
    y_ticks = np.arange(0, grid_size[1], 5)
    x_tick_labels = [(i + x_min) * 5 for i in x_ticks]
    y_tick_labels = [(i + y_min) * 5 for i in y_ticks]
    
    plt.xticks(x_ticks, x_tick_labels)
    plt.yticks(y_ticks, y_tick_labels)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved position heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_final_performance_comparison(algorithm_data, save_path=None):
    """
    Plot final performance metrics for all algorithms.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        save_path (str, optional): Path to save the figure
    """
    # Extract final performance metrics
    metrics = {
        'Final Reward': [],
        'Collision Rate': [],
        'Lane Invasion Rate': [],
        'Success Rate': [],
        'Junction Success': []
    }
    
    algorithms = []
    
    for algo, data in algorithm_data.items():
        algorithms.append(algo.upper())
        
        # Get final reward (average of last 10 episodes)
        if 'episode_rewards' in data and len(data['episode_rewards']) > 0:
            last_n = min(10, len(data['episode_rewards']))
            final_reward = np.mean(data['episode_rewards'][-last_n:])
            metrics['Final Reward'].append(final_reward)
        else:
            metrics['Final Reward'].append(0)
        
        # Get collision rate
        if 'episode_stats' in data and 'collision_count' in data['episode_stats']:
            collision_rate = np.mean(data['episode_stats']['collision_count'][-10:])
            metrics['Collision Rate'].append(collision_rate)
        else:
            metrics['Collision Rate'].append(0)
        
        # Get lane invasion rate
        if 'episode_stats' in data and 'lane_invasion_count' in data['episode_stats']:
            invasion_rate = np.mean(data['episode_stats']['lane_invasion_count'][-10:])
            metrics['Lane Invasion Rate'].append(invasion_rate)
        else:
            metrics['Lane Invasion Rate'].append(0)
        
        # Get success rate
        if 'episode_stats' in data and 'success_rate' in data['episode_stats']:
            success_rate = np.mean(data['episode_stats']['success_rate'][-10:])
            metrics['Success Rate'].append(success_rate)
        else:
            metrics['Success Rate'].append(0)
        
        # Get junction success rate
        if 'episode_stats' in data and 'junction_success_rate' in data['episode_stats']:
            junction_success = np.mean(data['episode_stats']['junction_success_rate'][-10:])
            metrics['Junction Success'].append(junction_success)
        else:
            metrics['Junction Success'].append(0)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
    
    for i, (metric, values) in enumerate(metrics.items()):
        # Plot bar chart
        bars = axes[i].bar(algorithms, values, color=[ALGORITHM_COLORS.get(algo.lower(), 'blue') for algo in algorithms])
        axes[i].set_title(f'{metric}', fontsize=14)
        axes[i].set_ylabel('Value', fontsize=12)
        axes[i].grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height * 1.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved final performance comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_computational_efficiency(algorithm_data, save_path=None):
    """
    Plot computational efficiency metrics for all algorithms.
    
    Args:
        algorithm_data (dict): Dictionary mapping algorithm names to metrics
        save_path (str, optional): Path to save the figure
    """
    # Extract computational metrics
    metrics = {
        'Training Time (s/episode)': [],
        'Inference Time (s/step)': [],
        'Memory Usage (GB)': []
    }
    
    algorithms = []
    
    for algo, data in algorithm_data.items():
        algorithms.append(algo.upper())
        
        if 'computational_metrics' in data:
            comp_metrics = data['computational_metrics']
            
            # Training time per episode
            if 'training_time_per_episode' in comp_metrics and comp_metrics['training_time_per_episode']:
                training_time = np.mean(comp_metrics['training_time_per_episode'])
                metrics['Training Time (s/episode)'].append(training_time)
            else:
                metrics['Training Time (s/episode)'].append(0)
            
            # Inference time per step
            if 'inference_time_per_step' in comp_metrics and comp_metrics['inference_time_per_step']:
                inference_time = np.mean(comp_metrics['inference_time_per_step'])
                metrics['Inference Time (s/step)'].append(inference_time)
            else:
                metrics['Inference Time (s/step)'].append(0)
            
            # Memory usage
            if 'memory_usage' in comp_metrics and comp_metrics['memory_usage']:
                memory_usage = np.mean(comp_metrics['memory_usage'])
                metrics['Memory Usage (GB)'].append(memory_usage)
            else:
                metrics['Memory Usage (GB)'].append(0)
        else:
            for metric in metrics:
                metrics[metric].append(0)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
    
    for i, (metric, values) in enumerate(metrics.items()):
        # Plot bar chart
        bars = axes[i].bar(algorithms, values, color=[ALGORITHM_COLORS.get(algo.lower(), 'blue') for algo in algorithms])
        axes[i].set_title(f'{metric}', fontsize=14)
        axes[i].set_ylabel('Value', fontsize=12)
        axes[i].grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height * 1.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved computational efficiency comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()

def generate_comparison_plots(algorithms, run_dirs, output_dir='plots'):
    """
    Generate all comparison plots for the specified algorithms.
    
    Args:
        algorithms (list): List of algorithm names
        run_dirs (dict): Dictionary mapping algorithm names to run directories
        output_dir (str): Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for each algorithm
    algorithm_data = {}
    for algo in algorithms:
        metrics_path = os.path.join(run_dirs[algo], f"{algo}_metrics.json")
        algo_data = load_metrics(metrics_path)
        algorithm_data[algo] = algo_data
    
    # Generate plots
    plot_reward_curves(algorithm_data, os.path.join(output_dir, 'reward_curves.png'))
    plot_learning_efficiency(algorithm_data, os.path.join(output_dir, 'learning_efficiency.png'))
    plot_loss_curves(algorithm_data, os.path.join(output_dir, 'loss_curves.png'))
    plot_centerline_deviation(algorithm_data, os.path.join(output_dir, 'centerline_deviation.png'))
    plot_collision_rates(algorithm_data, os.path.join(output_dir, 'collision_rates.png'))
    plot_action_distribution(algorithm_data, os.path.join(output_dir, 'action_distribution.png'))
    
    # Generate individual position heatmaps
    for algo in algorithms:
        plot_position_heatmap(algorithm_data, algo, os.path.join(output_dir, f'{algo}_position_heatmap.png'))
    
    # Generate final performance comparison
    plot_final_performance_comparison(algorithm_data, os.path.join(output_dir, 'final_performance.png'))
    
    # Generate computational efficiency comparison
    plot_computational_efficiency(algorithm_data, os.path.join(output_dir, 'computational_efficiency.png'))
    
    print(f"Generated all comparison plots in {output_dir}")

def plot_training_progress(metrics_paths, algo_names=None, output_dir=None):
    """
    Plot training progress for multiple runs or seeds of an algorithm.
    
    Args:
        metrics_paths (list): List of paths to metrics files
        algo_names (list, optional): List of names for each run
        output_dir (str, optional): Directory to save plots
    """
    if not metrics_paths:
        print("No metrics paths provided")
        return
    
    # Load data for each run
    run_data = []
    for i, path in enumerate(metrics_paths):
        data = load_metrics(path)
        if data:
            name = algo_names[i] if algo_names and i < len(algo_names) else f"Run {i+1}"
            run_data.append((name, data))
    
    if not run_data:
        print("No valid data found in provided metrics paths")
        return
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot reward curves
    plt.figure(figsize=(12, 8))
    
    for name, data in run_data:
        if 'episode_rewards' in data and len(data['episode_rewards']) > 0:
            rewards = data['episode_rewards']
            episodes = list(range(1, len(rewards) + 1))
            
            # Plot raw data with low alpha
            plt.plot(episodes, rewards, alpha=0.2)
            
            # Plot smoothed curve
            smoothed_rewards = smooth_curve(rewards, 0.9)
            plt.plot(episodes, smoothed_rewards, label=name, linewidth=2)
    
    plt.title('Episode Rewards During Training', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'reward_curves_by_run.png'), dpi=300, bbox_inches='tight')
        print(f"Saved reward curves by run to {os.path.join(output_dir, 'reward_curves_by_run.png')}")
    else:
        plt.show()
    
    plt.close()

def create_observation_figure(state, figsize=(8, 8)):
    """
    Create a figure to visualize the agent's observation.
    
    Args:
        state: The observation state (usually an image)
        figsize: Size of the figure
        
    Returns:
        fig: The created figure
    """
    fig = plt.figure(figsize=figsize)
    
    if len(state.shape) == 3:  # RGB image
        plt.imshow(state)
    elif len(state.shape) == 2:  # Grayscale image
        plt.imshow(state, cmap='gray')
    
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def plot_learning_curves(rewards, losses=None, smoothing=0.9, figsize=(12, 6)):
    """
    Plot learning curves for the agent.
    
    Args:
        rewards: List of episode rewards
        losses: List of losses
        smoothing: Exponential moving average smoothing factor
        figsize: Size of the figure
        
    Returns:
        fig: The created figure
    """
    fig = plt.figure(figsize=figsize)
    
    if losses is not None:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
    else:
        ax1 = plt.subplot(111)
    
    # Plot rewards
    episodes = np.arange(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, 'b-', alpha=0.3, label='Rewards')
    
    # Compute smoothed rewards
    smoothed_rewards = []
    if len(rewards) > 0:
        r_avg = rewards[0]
        for r in rewards:
            r_avg = smoothing * r_avg + (1 - smoothing) * r
            smoothed_rewards.append(r_avg)
        
        ax1.plot(episodes, smoothed_rewards, 'b-', label=f'Smoothed Rewards (α={smoothing})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot losses if provided
    if losses is not None and len(losses) > 0:
        loss_steps = np.arange(1, len(losses) + 1)
        ax2.plot(loss_steps, losses, 'r-', alpha=0.3, label='Loss')
        
        # Compute smoothed losses
        smoothed_losses = []
        loss_avg = losses[0]
        for loss in losses:
            loss_avg = smoothing * loss_avg + (1 - smoothing) * loss
            smoothed_losses.append(loss_avg)
            
        ax2.plot(loss_steps, smoothed_losses, 'r-', label=f'Smoothed Loss (α={smoothing})')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_all_metrics(algorithm_results, output_path=None):
    """
    Create a comprehensive plot of metrics for algorithm comparison.
    
    Args:
        algorithm_results: Dictionary of results for different algorithms
        output_path: Path to save the figure
    """
    if not algorithm_results:
        print("No results to plot")
        return
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    
    # Extract algorithm names
    algorithms = list(algorithm_results.keys())
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Plot reward comparison
    ax1 = plt.subplot(gs[0, 0])
    for i, algo in enumerate(algorithms):
        color = colors[i % len(colors)]
        if 'rewards' in algorithm_results[algo]:
            rewards = algorithm_results[algo]['rewards']
            mean_reward = np.mean(rewards)
            ax1.bar(i, mean_reward, color=color, label=f'{algo.upper()} ({mean_reward:.2f})')
    
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Algorithm Reward Comparison')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels([algo.upper() for algo in algorithms])
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot episode length comparison
    ax2 = plt.subplot(gs[0, 1])
    for i, algo in enumerate(algorithms):
        color = colors[i % len(colors)]
        if 'steps' in algorithm_results[algo]:
            steps = algorithm_results[algo]['steps']
            mean_steps = np.mean(steps)
            ax2.bar(i, mean_steps, color=color, label=f'{algo.upper()} ({mean_steps:.2f})')
    
    ax2.set_ylabel('Average Episode Length')
    ax2.set_title('Episode Length Comparison')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels([algo.upper() for algo in algorithms])
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Plot safety metrics
    ax3 = plt.subplot(gs[1, 0])
    width = 0.3
    x = np.arange(len(algorithms))
    
    for i, algo in enumerate(algorithms):
        collision_count = algorithm_results[algo].get('collision_count', 0)
        lane_invasion_count = algorithm_results[algo].get('lane_invasion_count', 0)
        
        ax3.bar(x[i] - width/2, collision_count, width, color='r', label='Collisions' if i == 0 else "")
        ax3.bar(x[i] + width/2, lane_invasion_count, width, color='y', label='Lane Invasions' if i == 0 else "")
    
    ax3.set_ylabel('Count')
    ax3.set_title('Safety Metrics')
    ax3.set_xticks(x)
    ax3.set_xticklabels([algo.upper() for algo in algorithms])
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Plot centerline deviations
    ax4 = plt.subplot(gs[1, 1])
    for i, algo in enumerate(algorithms):
        color = colors[i % len(colors)]
        if 'centerline_deviations' in algorithm_results[algo]:
            centerline_dev = algorithm_results[algo]['centerline_deviations']
            if centerline_dev:
                mean_dev = np.mean(centerline_dev)
                ax4.bar(i, mean_dev, color=color, label=f'{algo.upper()} ({mean_dev:.2f})')
    
    ax4.set_ylabel('Average Centerline Deviation (m)')
    ax4.set_title('Path Following Accuracy')
    ax4.set_xticks(range(len(algorithms)))
    ax4.set_xticklabels([algo.upper() for algo in algorithms])
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    return fig

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization tool for CARLA RL experiments")
    parser.add_argument("--mode", choices=["compare", "training"], default="compare",
                      help="Visualization mode: compare algorithms or show training progress")
    parser.add_argument("--algorithms", nargs="+", default=["dqn", "ddqn", "dueling_dqn", "sarsa", "ppo"],
                      help="List of algorithms to compare")
    parser.add_argument("--run_dirs", nargs="+", 
                      help="Directories containing algorithm runs (one per algorithm)")
    parser.add_argument("--metrics_paths", nargs="+",
                      help="Paths to metrics files for training progress visualization")
    parser.add_argument("--output_dir", default="plots",
                      help="Directory to save plots")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        if not args.run_dirs or len(args.run_dirs) != len(args.algorithms):
            print("Error: Must provide one run directory for each algorithm")
        else:
            run_dirs = {algo: dir for algo, dir in zip(args.algorithms, args.run_dirs)}
            generate_comparison_plots(args.algorithms, run_dirs, args.output_dir)
            
    elif args.mode == "training":
        if not args.metrics_paths:
            print("Error: Must provide at least one metrics path")
        else:
            plot_training_progress(args.metrics_paths, args.algorithms, args.output_dir) 