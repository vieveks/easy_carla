"""
Base Agent class for all RL algorithms
"""
import os
import torch
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import sys

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import ensure_directory, save_metrics, load_metrics, preprocess_image

class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    """
    
    def __init__(self, 
                 state_shape, 
                 action_size, 
                 device=None,
                 model_dir='models',
                 tensorboard_dir='logs'):
        """
        Initialize the agent.
        
        Args:
            state_shape (tuple): Shape of the state space
            action_size (int): Size of the action space
            device (str, optional): Device to use for computation ('cpu' or 'cuda')
            model_dir (str): Directory to save models
            tensorboard_dir (str): Directory for TensorBoard logs
        """
        self.state_shape = state_shape
        self.action_size = action_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Directories
        self.model_dir = model_dir
        self.tensorboard_dir = tensorboard_dir
        
        # Create directories
        ensure_directory(self.model_dir)
        ensure_directory(self.tensorboard_dir)
        
        # Initialize models
        self._init_models()
        
        # Metrics for logging
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'evaluation_rewards': [],
            'centerline_deviations': [],
            'junction_actions': {
                'left': 0,
                'right': 0,
                'forward': 0,
                'other': 0
            },
            # Extended metrics
            'action_distribution': {str(i): 0 for i in range(self.action_size)},
            'position_heatmap': {},  # Will be filled with position keys
            'speeds': [],  # Track speeds throughout training
            'reward_components': {  # Track individual reward components
                'forward': [],
                'collision': [],
                'lane_invasion': [],
                'centerline': [],
                'speed': [],
                'time': []
            },
            'episode_stats': {  # Per-episode statistics
                'collision_count': [],
                'lane_invasion_count': [],
                'avg_centerline_deviation': [],
                'avg_speed': [],
                'progress': [],
                'success_rate': [],
                'junction_success_rate': []
            },
            'computational_metrics': {  # Computational efficiency
                'training_time_per_episode': [],
                'inference_time_per_step': [],
                'memory_usage': []
            }
        }
        
        # Episode-specific variables
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_loss = 0
        self.training_step = 0
        
        # Tracking for current episode
        self.episode_start_time = None
        self.episode_speeds = []
        self.episode_actions = []
        self.episode_reward_components = {
            'forward': 0,
            'collision': 0,
            'lane_invasion': 0,
            'centerline': 0,
            'speed': 0,
            'time': 0
        }
    
    @abstractmethod
    def _init_models(self):
        """
        Initialize neural network models.
        This method should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def select_action(self, state, training=True):
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            training (bool): Whether the agent is in training mode
            
        Returns:
            int: Selected action
        """
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done (bool): Whether the episode is done
            
        Returns:
            float: Loss value (if any)
        """
        pass
    
    def preprocess_state(self, state):
        """
        Preprocess the state for neural network input.
        
        Args:
            state: Raw state from the environment
            
        Returns:
            torch.Tensor: Processed state tensor
        """
        # If state is an image, preprocess it
        if isinstance(state, np.ndarray) and len(state.shape) == 3:
            return preprocess_image(state).to(self.device)
        
        # If state is already a tensor, ensure it's on the right device
        if isinstance(state, torch.Tensor):
            return state.to(self.device)
        
        # Otherwise, convert to tensor
        return torch.tensor(state, dtype=torch.float32).to(self.device)
    
    def start_episode(self):
        """
        Called at the beginning of an episode.
        """
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_loss = 0
        
        # Tracking for current episode
        self.episode_start_time = time.time()
        self.episode_speeds = []
        self.episode_actions = []
        self.episode_reward_components = {
            'forward': 0,
            'collision': 0,
            'lane_invasion': 0,
            'centerline': 0,
            'speed': 0,
            'time': 0
        }
    
    def end_episode(self):
        """
        Called at the end of an episode.
        Updates metrics and logs information.
        """
        # Record metrics
        self.metrics['episode_rewards'].append(self.current_episode_reward)
        self.metrics['episode_lengths'].append(self.current_episode_length)
        
        if self.current_episode_length > 0:
            avg_loss = self.current_episode_loss / self.current_episode_length
        else:
            avg_loss = 0
            
        self.metrics['losses'].append(avg_loss)
        
        # Reset episode-specific variables
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_loss = 0
        
        # Calculate episode-level metrics
        episode_duration = time.time() - self.episode_start_time
        self.episode_speeds.append(self.current_episode_reward / episode_duration)
        self.episode_actions.append(self.current_episode_length)
        self.episode_reward_components['time'] += episode_duration
        
        # Update episode-level reward components
        self.episode_reward_components['forward'] += self.current_episode_reward
        
        # Update episode-level metrics
        self.metrics['episode_stats']['collision_count'].append(self.episode_reward_components['collision'])
        self.metrics['episode_stats']['lane_invasion_count'].append(self.episode_reward_components['lane_invasion'])
        self.metrics['episode_stats']['avg_centerline_deviation'].append(self.metrics['centerline_deviations'][-1] if self.metrics['centerline_deviations'] else 0)
        self.metrics['episode_stats']['avg_speed'].append(self.episode_speeds[-1])
        self.metrics['episode_stats']['progress'].append(self.current_episode_length / self.current_episode_length)
        self.metrics['episode_stats']['success_rate'].append(self.current_episode_reward > 0)
        self.metrics['episode_stats']['junction_success_rate'].append(self.episode_reward_components['forward'] > 0)
        
        # Update metrics
        self.metrics['evaluation_rewards'].append(self.current_episode_reward)
        
        # Update reward components
        self.episode_reward_components['collision'] = 0
        self.episode_reward_components['lane_invasion'] = 0
        
        # Update position heatmap
        position_key = f"{self.current_episode_length // 10}-{self.current_episode_length % 10}"
        if position_key not in self.metrics['position_heatmap']:
            self.metrics['position_heatmap'][position_key] = 0
        self.metrics['position_heatmap'][position_key] += 1
        
        # Update action distribution
        for action in self.episode_actions[-self.current_episode_length:]:
            self.metrics['action_distribution'][str(action)] += 1
        
        # Update reward components
        self.metrics['reward_components']['forward'].append(self.episode_reward_components['forward'])
        self.metrics['reward_components']['collision'].append(self.episode_reward_components['collision'])
        self.metrics['reward_components']['lane_invasion'].append(self.episode_reward_components['lane_invasion'])
        self.metrics['reward_components']['centerline'].append(self.episode_reward_components['centerline'])
        self.metrics['reward_components']['speed'].append(self.episode_reward_components['speed'])
        self.metrics['reward_components']['time'].append(self.episode_reward_components['time'])
        
        # Update computational metrics
        self.metrics['computational_metrics']['training_time_per_episode'].append(episode_duration)
        self.metrics['computational_metrics']['inference_time_per_step'].append(episode_duration / self.current_episode_length)
        self.metrics['computational_metrics']['memory_usage'].append(torch.cuda.memory_allocated() / 1024**3)
    
    def update_episode_stats(self, reward, loss=None):
        """
        Update episode statistics.
        
        Args:
            reward (float): Reward received in the current step
            loss (float, optional): Loss value from the current update
        """
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.training_step += 1
        
        if loss is not None:
            self.current_episode_loss += loss
    
    def update_extended_metrics(self, state, action, reward, info=None):
        """
        Update extended metrics for more detailed analysis.
        
        Args:
            state: Current state
            action (int): Action taken
            reward (float): Reward received
            info (dict, optional): Additional information from the environment
        """
        # Track actions taken
        self.episode_actions.append(action)
        self.metrics['action_distribution'][str(action)] += 1
        
        # Track position if available in info
        if info and 'vehicle_state' in info:
            loc = info['vehicle_state']['location']
            position_key = f"{int(loc['x'])//5}-{int(loc['y'])//5}"
            if position_key not in self.metrics['position_heatmap']:
                self.metrics['position_heatmap'][position_key] = 0
            self.metrics['position_heatmap'][position_key] += 1
        
        # Track speed if available in info
        if info and 'vehicle_state' in info:
            speed = info['vehicle_state']['speed']
            self.episode_speeds.append(speed)
            self.metrics['speeds'].append(speed)
            self.episode_reward_components['speed'] += 0.2 * min(speed, 30) / 30.0
        
        # Track reward components if available in info
        if info:
            if 'move_reward' in info:
                self.episode_reward_components['forward'] += info['move_reward']
            if 'collision' in info and info['collision']:
                self.episode_reward_components['collision'] += 1
            if 'lane_invasion' in info and info['lane_invasion']:
                self.episode_reward_components['lane_invasion'] += 1
            if 'centerline_penalty' in info:
                self.episode_reward_components['centerline'] += info['centerline_penalty']
            if 'time_penalty' in info:
                self.episode_reward_components['time'] += info['time_penalty']
    
    def log_junction_action(self, action):
        """
        Log an action taken at a junction.
        
        Args:
            action (int): Action taken
        """
        if action == 2:  # Left
            self.metrics['junction_actions']['left'] += 1
        elif action == 3:  # Right
            self.metrics['junction_actions']['right'] += 1
        elif action == 1:  # Forward
            self.metrics['junction_actions']['forward'] += 1
        else:
            self.metrics['junction_actions']['other'] += 1
        
        # Update success rate at junctions
        if hasattr(self, 'metrics') and 'episode_stats' in self.metrics:
            success = action in [1, 2, 3]  # Consider left, right, forward as success
            if len(self.metrics['episode_stats']['junction_success_rate']) > 0:
                # Update the most recent success rate (for current episode)
                self.metrics['episode_stats']['junction_success_rate'][-1] = success
    
    def log_centerline_deviation(self, deviation):
        """
        Log the deviation from the centerline.
        
        Args:
            deviation (float): Distance from centerline
        """
        self.metrics['centerline_deviations'].append(deviation)
    
    def save(self, file_path=None):
        """
        Save the agent's models and metrics.
        
        Args:
            file_path (str, optional): Path to save the model. 
                                      If None, use default naming.
        """
        # Implement in subclasses
        pass
    
    def load(self, file_path):
        """
        Load agent's models and metrics.
        
        Args:
            file_path (str): Path to the saved model
        """
        # Implement in subclasses
        pass
    
    def save_metrics(self, file_path=None):
        """
        Save training metrics to a file.
        
        Args:
            file_path (str, optional): Path to save metrics.
                                      If None, use default naming.
        """
        if file_path is None:
            file_path = os.path.join(self.model_dir, f"{self.__class__.__name__}_metrics.json")
        
        # Save metrics to file
        save_metrics(self.metrics, file_path)
    
    def load_metrics(self, file_path):
        """
        Load metrics from a file.
        
        Args:
            file_path (str): Path to the metrics file
        """
        # Load metrics from file
        loaded_metrics = load_metrics(file_path)
        
        # Update current metrics
        self.metrics.update(loaded_metrics)
    
    def evaluate(self, env, num_episodes=5):
        """
        Evaluate the agent on the environment.
        
        Args:
            env: Environment to evaluate on
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        evaluation_rewards = []
        evaluation_lengths = []
        
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Select action without exploration
                action = self.select_action(state, training=False)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                
                # Update state
                state = next_state
            
            # Record metrics
            evaluation_rewards.append(episode_reward)
            evaluation_lengths.append(episode_length)
        
        # Calculate statistics
        avg_reward = sum(evaluation_rewards) / num_episodes
        avg_length = sum(evaluation_lengths) / num_episodes
        
        # Update metrics
        self.metrics['evaluation_rewards'].append(avg_reward)
        
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'rewards': evaluation_rewards,
            'lengths': evaluation_lengths
        }
    
    def save_results(self, file_path):
        """
        Save training results to a file.
        
        Args:
            file_path (str): Path to save results
        """
        # Create a results dictionary
        results = {
            'episode_rewards': self.metrics['episode_rewards'],
            'episode_lengths': self.metrics['episode_lengths'],
            'losses': self.metrics['losses'],
        }
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    def save_all_metrics(self, file_path):
        """
        Save all metrics to a JSON file for visualization.
        
        Args:
            file_path (str): Path to save metrics
        """
        # Save all metrics - serialize to JSON safe format
        metrics_to_save = self.metrics.copy()
        
        # Handle special data structures (numpy arrays, etc.)
        for key, value in metrics_to_save.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list) and subvalue and isinstance(subvalue[0], np.ndarray):
                        metrics_to_save[key][subkey] = [x.tolist() if isinstance(x, np.ndarray) else x for x in subvalue]
                    elif isinstance(subvalue, np.ndarray):
                        metrics_to_save[key][subkey] = subvalue.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                metrics_to_save[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in value]
            elif isinstance(value, np.ndarray):
                metrics_to_save[key] = value.tolist()
        
        # Handle non-serializable types
        for key in ['episode_speeds', 'episode_actions', 'episode_reward_components']:
            if hasattr(self, key):
                metrics_to_save[key] = getattr(self, key)
                
                # Convert numpy types to native types
                if isinstance(metrics_to_save[key], list) and metrics_to_save[key] and isinstance(metrics_to_save[key][0], np.ndarray):
                    metrics_to_save[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in metrics_to_save[key]]
                elif isinstance(metrics_to_save[key], np.ndarray):
                    metrics_to_save[key] = metrics_to_save[key].tolist()
                elif isinstance(metrics_to_save[key], dict):
                    for subkey, subvalue in metrics_to_save[key].items():
                        if isinstance(subvalue, np.ndarray):
                            metrics_to_save[key][subkey] = subvalue.tolist()
                        elif isinstance(subvalue, list) and subvalue and isinstance(subvalue[0], np.ndarray):
                            metrics_to_save[key][subkey] = [x.tolist() if isinstance(x, np.ndarray) else x 
                                                          for x in subvalue]
        
        # Add algorithm name and timestamp
        metrics_to_save['algorithm'] = self.__class__.__name__.replace('Agent', '').lower()
        metrics_to_save['timestamp'] = time.strftime("%Y%m%d-%H%M%S")
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4) 