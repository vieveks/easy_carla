"""
Helper functions for the CARLA RL project
"""
import os
import json
import torch
import numpy as np
import random
import datetime
import shutil
import time
from typing import Dict, List, Tuple, Any, Optional, Union

def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_directory(directory_path):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

def save_config(config_dict, save_path):
    """
    Save configuration to a JSON file.
    
    Args:
        config_dict (dict): Configuration dictionary
        save_path (str): Path to save the JSON file
    """
    # Ensure directory exists
    ensure_directory(os.path.dirname(save_path))
    
    # Convert non-serializable objects to strings
    for key, value in config_dict.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            config_dict[key] = str(value)
            
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the JSON file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def save_model(model, optimizer, save_path, additional_info=None):
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        save_path (str): Path to save the checkpoint
        additional_info (dict, optional): Additional information to save
    """
    # Ensure directory exists
    ensure_directory(os.path.dirname(save_path))
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Add additional info if provided
    if additional_info:
        checkpoint.update(additional_info)
    
    # Save checkpoint
    torch.save(checkpoint, save_path)

def load_model(model, optimizer, load_path):
    """
    Load model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load weights into
        optimizer (torch.optim.Optimizer): The optimizer to load state into
        load_path (str): Path to the checkpoint file
        
    Returns:
        tuple: (model, optimizer, additional_info)
    """
    checkpoint = torch.load(load_path)
    
    # Load model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Remove standard keys to get additional info
    additional_info = {k: v for k, v in checkpoint.items() 
                      if k not in ['model_state_dict', 'optimizer_state_dict', 'timestamp']}
    
    return model, optimizer, additional_info

def create_experiment_dir(base_dir, experiment_name=None):
    """
    Create a directory for the experiment with timestamp.
    
    Args:
        base_dir (str): Base directory for experiments
        experiment_name (str, optional): Name of the experiment
        
    Returns:
        str: Path to the created experiment directory
    """
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory name
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
    
    # Create full path
    experiment_dir = os.path.join(base_dir, dir_name)
    
    # Create directory
    ensure_directory(experiment_dir)
    
    return experiment_dir

def save_metrics(metrics, save_path):
    """
    Save training/evaluation metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary of metrics
        save_path (str): Path to save the JSON file
    """
    # Ensure directory exists
    ensure_directory(os.path.dirname(save_path))
    
    # Convert numpy arrays to lists for JSON serialization
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            metrics[key] = [v.tolist() for v in value]
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(load_path):
    """
    Load metrics from a JSON file.
    
    Args:
        load_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary of metrics
    """
    with open(load_path, 'r') as f:
        return json.load(f)

def preprocess_image(image):
    """
    Preprocess image for neural network input.
    
    Args:
        image (numpy.ndarray): RGB image array (H, W, C) or batch of images (B, H, W, C)
        
    Returns:
        torch.Tensor: Processed image tensor (C, H, W) or (B, C, H, W)
    """
    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0
    
    # Check if it's a batch or single image
    if len(image.shape) == 4:  # Batch of images (B, H, W, C)
        # Convert to PyTorch tensor and change dimension order to (B, C, H, W)
        image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).contiguous()
    else:  # Single image (H, W, C)
        # Convert to PyTorch tensor and change dimension order to (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        # Add batch dimension if it's a single image
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Ensure tensor is contiguous
        image_tensor = image_tensor.contiguous()
    
    return image_tensor

def update_progress_bar(progress_bar, episode, total_episodes, episode_reward, 
                       episode_loss, additional_info=None):
    """
    Update the progress bar with current training information.
    
    Args:
        progress_bar: tqdm progress bar instance
        episode (int): Current episode number
        total_episodes (int): Total number of episodes
        episode_reward (float): Reward for the current episode
        episode_loss (float): Loss for the current episode
        additional_info (dict, optional): Additional information to display
    """
    # Update progress
    progress_bar.update(1)
    
    # Prepare postfix info
    postfix = {
        'reward': f"{episode_reward:.2f}",
        'loss': f"{episode_loss:.4f}",
    }
    
    # Add additional info if provided
    if additional_info:
        postfix.update(additional_info)
    
    # Update progress bar
    progress_bar.set_postfix(postfix)

def get_epsilon_greedy_action(q_values, epsilon, action_space_size):
    """
    Select action using epsilon-greedy policy.
    
    Args:
        q_values (torch.Tensor): Q-values for each action
        epsilon (float): Exploration rate (0-1)
        action_space_size (int): Number of possible actions
        
    Returns:
        int: Selected action
    """
    # With probability epsilon, select random action
    if random.random() < epsilon:
        return random.randint(0, action_space_size - 1)
    
    # Otherwise, select action with highest Q-value
    return q_values.argmax().item()

def calculate_epsilon(step, epsilon_start, epsilon_end, epsilon_decay):
    """
    Calculate epsilon value for epsilon-greedy exploration.
    
    Args:
        step (int): Current time step
        epsilon_start (float): Starting epsilon value
        epsilon_end (float): Final epsilon value
        epsilon_decay (int): Number of steps for decay
        
    Returns:
        float: Current epsilon value
    """
    # Linear decay
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
              max(0, (epsilon_decay - step)) / epsilon_decay
    
    return epsilon

def calculate_returns(rewards, gamma=0.99):
    """
    Calculate discounted returns for a sequence of rewards.
    
    Args:
        rewards (list): List of rewards
        gamma (float): Discount factor
        
    Returns:
        list: Discounted returns
    """
    returns = []
    R = 0
    
    # Calculate returns in reverse order
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    return returns 