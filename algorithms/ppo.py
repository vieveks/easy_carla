"""
Proximal Policy Optimization (PPO) implementation
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time
import sys
import json

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.base_agent import BaseAgent
from utils.helpers import save_model, load_model, calculate_returns

class PPOModel(nn.Module):
    """
    Neural network model for PPO.
    
    This network outputs both a policy (action probabilities) and a value function.
    """
    
    def __init__(self, state_shape, action_size, hidden_dim=512):
        """
        Initialize the model.
        
        Args:
            state_shape (tuple): Shape of the state (C, H, W)
            action_size (int): Number of possible actions
            hidden_dim (int): Size of hidden layers
        """
        super(PPOModel, self).__init__()
        
        # Extract dimensions from state shape
        channels, height, width = state_shape
        
        # CNN layers for feature extraction (shared)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate size of CNN output
        # After CNN, the image size is reduced based on kernel sizes and strides
        # Formula: output_size = (input_size - kernel_size) / stride + 1
        conv1_out_h = (height - 8) // 4 + 1
        conv1_out_w = (width - 8) // 4 + 1
        
        conv2_out_h = (conv1_out_h - 4) // 2 + 1
        conv2_out_w = (conv1_out_w - 4) // 2 + 1
        
        conv3_out_h = (conv2_out_h - 3) // 1 + 1
        conv3_out_w = (conv2_out_w - 3) // 1 + 1
        
        cnn_output_size = 64 * conv3_out_h * conv3_out_w
        
        # Shared features layer
        self.features = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): Input state (batch_size, C, H, W)
            
        Returns:
            tuple: (action_probs, state_value)
        """
        # Pass through CNN
        x = self.cnn(state)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared features
        features = self.features(x)
        
        # Get action probabilities and state value
        action_probs = self.policy(features)
        state_value = self.value(features)
        
        return action_probs, state_value

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    
    This implementation uses the clipped surrogate objective
    and alternates between collecting trajectories and optimization.
    """
    
    def __init__(self, 
                 state_shape=(3, 84, 84), 
                 action_size=7,
                 hidden_dim=512,
                 learning_rate=1e-4,
                 gamma=0.99,
                 clip_param=0.2,
                 ppo_epochs=10,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 batch_size=64,
                 device=None,
                 model_dir='models',
                 tensorboard_dir='logs'):
        """
        Initialize the PPO agent.
        
        Args:
            state_shape (tuple): Shape of the state (C, H, W)
            action_size (int): Number of possible actions
            hidden_dim (int): Size of hidden layers
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor for future rewards
            clip_param (float): PPO clipping parameter
            ppo_epochs (int): Number of epochs to optimize on the same data
            value_loss_coef (float): Value loss coefficient
            entropy_coef (float): Entropy coefficient for encouraging exploration
            max_grad_norm (float): Maximum norm for gradient clipping
            batch_size (int): Batch size for training
            device (str, optional): Device to use for computation
            model_dir (str): Directory to save models
            tensorboard_dir (str): Directory for TensorBoard logs
        """
        # Store parameters as instance variables before calling parent constructor
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        # Initialize base agent
        super(PPOAgent, self).__init__(
            state_shape=state_shape,
            action_size=action_size,
            device=device,
            model_dir=model_dir,
            tensorboard_dir=tensorboard_dir
        )
        
        # Memory for storing trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def _init_models(self):
        """
        Initialize the PPO model.
        """
        self.model = PPOModel(self.state_shape, self.action_size, self.hidden_dim).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def select_action(self, state, training=True, eval_mode=False):
        """
        Select an action using the current policy.
        
        Args:
            state: Current state
            training (bool): Whether the agent is in training mode
            eval_mode (bool): Whether we're in evaluation mode
            
        Returns:
            int: Selected action
        """
        # Preprocess state
        state_tensor = self.preprocess_state(state)
        
        # Get action probabilities and state value
        with torch.no_grad():
            action_probs, state_value = self.model(state_tensor)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        # Sample action
        if not eval_mode and training:
            action = dist.sample()
        else:
            # During evaluation, take the most probable action
            action = action_probs.argmax(dim=-1)
        
        # Store trajectory information if training
        if training and not eval_mode:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(dist.log_prob(action).item())
            self.values.append(state_value.item())
        
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        """
        Legacy method that is kept for compatibility with the BaseAgent interface.
        In the current main loop, we use store_experience and train_step instead.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done (bool): Whether the episode is done
            
        Returns:
            float: Loss value if update was performed, 0 otherwise
        """
        # For compatibility, just call store_experience
        self.store_experience(state, action, reward, next_state, done)
        
        # Only update at the end of an episode
        if done:
            return self._update_policy()
        
        return 0.0
    
    def _update_policy(self):
        """
        Update policy using PPO algorithm.
        
        Returns:
            float: Average policy loss
        """
        # Convert stored trajectories to tensors
        states = [self.preprocess_state(s) for s in self.states]
        states = torch.cat(states, dim=0).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)
        
        # Calculate returns and advantages
        returns = self._compute_returns(rewards, dones)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimization loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Create dataset from memory
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Perform multiple optimization epochs
        for _ in range(self.ppo_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                # Forward pass
                action_probs, values = self.model(batch_states)
                dist = Categorical(action_probs)
                
                # Get new log probabilities
                new_log_probs = dist.log_prob(batch_actions)
                
                # Compute ratio (π_θ / π_θ_old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                
                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                # Calculate entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Record losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        # Calculate average losses
        num_batches = len(dataloader) * self.ppo_epochs
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy = total_entropy / num_batches
        
        # Total loss (for reporting)
        total_loss = avg_policy_loss + self.value_loss_coef * avg_value_loss - self.entropy_coef * avg_entropy
        
        return total_loss
    
    def _compute_returns(self, rewards, dones):
        """
        Compute discounted returns.
        
        Args:
            rewards (torch.Tensor): Tensor of rewards
            dones (torch.Tensor): Tensor of done flags
            
        Returns:
            torch.Tensor: Discounted returns
        """
        # Calculate returns using Generalized Advantage Estimation
        returns = torch.zeros_like(rewards)
        next_return = 0
        
        for t in reversed(range(len(rewards))):
            # If episode ended, start with 0
            next_return = 0 if dones[t] else next_return
            
            # Compute return
            returns[t] = rewards[t] + self.gamma * next_return * (1 - dones[t])
            next_return = returns[t]
        
        return returns
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience for PPO.
        
        This method is required for compatibility with the main training loop,
        but most of the storage happens in select_action for PPO.
        
        Args:
            state: Current state (already stored during select_action)
            action: Action taken (already stored during select_action)
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Store reward and done flag
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train_step(self):
        """
        Perform a single training step.
        Only update at the end of an episode.
        
        Returns:
            float: Loss value if update was performed, 0 otherwise
        """
        if len(self.rewards) > 0 and self.dones[-1]:
            return self._update_policy()
        return 0.0
    
    def save_metrics(self, episode, reward, loss):
        """
        Save performance metrics for the current episode.
        
        Args:
            episode (int): Episode number
            reward (float): Episode reward
            loss (float): Average loss for the episode
        """
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = {
                'episodes': [],
                'rewards': [],
                'losses': []
            }
        
        self.metrics_history['episodes'].append(episode)
        self.metrics_history['rewards'].append(reward)
        self.metrics_history['losses'].append(loss)
    
    def save_additional_metrics(self, episode, stats):
        """
        Save additional metrics from episode statistics.
        
        Args:
            episode (int): Episode number
            stats (dict): Episode statistics
        """
        if not hasattr(self, 'additional_metrics'):
            self.additional_metrics = {
                'centerline_deviations': [],
                'junction_actions': {
                    'left': 0,
                    'right': 0,
                    'forward': 0,
                    'other': 0
                }
            }
        
        # Save centerline deviation
        if 'avg_centerline_deviation' in stats:
            self.additional_metrics['centerline_deviations'].append(stats['avg_centerline_deviation'])
        
        # Update junction actions
        if 'junction_actions' in stats:
            for action_type, count in stats['junction_actions'].items():
                if action_type in self.additional_metrics['junction_actions']:
                    self.additional_metrics['junction_actions'][action_type] += count
    
    def save(self, file_path=None):
        """
        Save agent's models and metrics.
        
        Args:
            file_path (str, optional): Path to save the model.
                                      If None, use default naming.
        """
        if file_path is None:
            file_path = os.path.join(self.model_dir, 'ppo_model.pth')
        
        # Save model, optimizer, and additional info
        additional_info = {
            'training_step': self.training_step,
            'gamma': self.gamma,
            'clip_param': self.clip_param,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef
        }
        
        save_model(self.model, self.optimizer, file_path, additional_info)
        
        # Save metrics to a separate file using the parent class method
        metrics_path = os.path.join(os.path.dirname(file_path), 'PPOAgent_metrics.json')
        super().save_metrics(metrics_path)
    
    def load(self, file_path):
        """
        Load agent's models and metrics.
        
        Args:
            file_path (str): Path to the saved model
        """
        # Load model, optimizer, and additional info
        self.model, self.optimizer, additional_info = load_model(
            self.model,
            self.optimizer,
            file_path
        )
        
        # Update agent parameters
        self.training_step = additional_info.get('training_step', 0)
        self.gamma = additional_info.get('gamma', self.gamma)
        self.clip_param = additional_info.get('clip_param', self.clip_param)
        self.value_loss_coef = additional_info.get('value_loss_coef', self.value_loss_coef)
        self.entropy_coef = additional_info.get('entropy_coef', self.entropy_coef)
        
        # Load metrics
        metrics_path = os.path.join(os.path.dirname(file_path), 'PPOAgent_metrics.json')
        if os.path.exists(metrics_path):
            self.load_metrics(metrics_path)
    
    def plot_learning_curves(self, file_path):
        """
        Plot learning curves for the agent.
        
        Args:
            file_path (str): Path to save the plot
        """
        if not hasattr(self, 'metrics_history'):
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure with 2 subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        
        # Plot rewards
        episodes = self.metrics_history['episodes']
        rewards = self.metrics_history['rewards']
        axs[0].plot(episodes, rewards, label='Episode Reward', alpha=0.7)
        
        # Add a running average line for rewards
        if len(rewards) > 10:  # Only add if we have enough data points
            window_size = min(10, len(rewards) // 5)  # Use 10 or 20% of total episodes
            running_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            running_avg_episodes = episodes[window_size-1:]
            axs[0].plot(running_avg_episodes, running_avg, 'r-', label=f'{window_size}-Episode Average', linewidth=2)
        
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot losses with log scale
        losses = self.metrics_history['losses']
        axs[1].plot(episodes, losses)
        axs[1].set_title('Training Loss')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Loss (log scale)')
        axs[1].set_yscale('log')  # Use logarithmic scale for loss values
        axs[1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
            
    def save_results(self, file_path):
        """
        Save all results to a JSON file.
        
        Args:
            file_path (str): Path to save the results
        """
        results = {}
        
        # Save metrics history
        if hasattr(self, 'metrics_history'):
            results['metrics'] = self.metrics_history
        
        # Save additional metrics
        if hasattr(self, 'additional_metrics'):
            results['additional_metrics'] = self.additional_metrics
        
        # Save hyperparameters
        results['hyperparameters'] = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'clip_param': self.clip_param, 
            'ppo_epochs': self.ppo_epochs,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'batch_size': self.batch_size
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4) 