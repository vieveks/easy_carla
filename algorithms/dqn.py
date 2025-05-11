"""
Deep Q-Network (DQN) implementation
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from collections import deque
import sys
import json

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
from utils.helpers import calculate_epsilon, get_epsilon_greedy_action, save_model, load_model

class DQNModel(nn.Module):
    """
    Neural network model for DQN.
    
    This network maps an image state to Q-values for each action.
    """
    
    def __init__(self, state_shape, action_size, hidden_dim=512):
        """
        Initialize the model.
        
        Args:
            state_shape (tuple): Shape of the state (C, H, W)
            action_size (int): Number of possible actions
            hidden_dim (int): Size of hidden layers
        """
        super(DQNModel, self).__init__()
        
        # Extract dimensions from state shape
        channels, height, width = state_shape
        
        # CNN layers for feature extraction
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
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): Input state (batch_size, C, H, W)
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Pass through CNN
        features = self.cnn(state)
        
        # Flatten - using reshape instead of view to handle non-contiguous tensors
        features = features.reshape(features.size(0), -1)
        
        # Pass through fully connected layers
        q_values = self.fc(features)
        
        return q_values

class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent.
    
    This implementation follows the original DQN algorithm with experience replay
    and target network for stable learning.
    """
    
    def __init__(self, 
                 state_shape=(3, 84, 84), 
                 action_size=7,
                 hidden_dim=512,
                 learning_rate=5e-5,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=50000,
                 target_update=10,
                 memory_size=100000,
                 batch_size=64,
                 device=None,
                 model_dir='models',
                 tensorboard_dir='logs'):
        """
        Initialize the DQN agent.
        
        Args:
            state_shape (tuple): Shape of the state (C, H, W)
            action_size (int): Number of possible actions
            hidden_dim (int): Size of hidden layers
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor for future rewards
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (int): Number of steps for epsilon decay
            target_update (int): Frequency of target network update
            memory_size (int): Maximum size of replay buffer
            batch_size (int): Batch size for training
            device (str, optional): Device to use for computation
            model_dir (str): Directory to save models
            tensorboard_dir (str): Directory for TensorBoard logs
        """
        # Initialize base agent
        super(DQNAgent, self).__init__(
            state_shape=state_shape,
            action_size=action_size,
            device=device,
            model_dir=model_dir,
            tensorboard_dir=tensorboard_dir
        )
        
        # Set hyperparameters
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        
        # Create replay buffer
        self.memory = ReplayBuffer(memory_size, batch_size)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize loss function - use Huber loss for stability instead of MSE
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss is more robust to outliers
    
    def _init_models(self):
        """
        Initialize Q-network and target network.
        """
        hidden_dim = getattr(self, 'hidden_dim', 512)  # Use default value if not set
        self.q_network = DQNModel(self.state_shape, self.action_size, hidden_dim).to(self.device)
        self.target_network = DQNModel(self.state_shape, self.action_size, hidden_dim).to(self.device)
        
        # Initialize target network with same weights as Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
    
    def select_action(self, state, training=True, eval_mode=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training (bool): Whether the agent is in training mode
            eval_mode (bool): Whether the agent is in evaluation mode
            
        Returns:
            int: Selected action
        """
        # Preprocess state properly for channel ordering
        if isinstance(state, np.ndarray) and len(state.shape) == 3 and state.shape[2] == 3:
            # Convert HWC to CHW format
            state_tensor = torch.from_numpy(state.astype(np.float32) / 255.0)
            state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        else:
            state_tensor = self.preprocess_state(state).contiguous()
        
        # Calculate epsilon for exploration
        if training and not eval_mode:
            epsilon = calculate_epsilon(
                self.training_step,
                self.epsilon_start,
                self.epsilon_end,
                self.epsilon_decay
            )
        else:
            epsilon = 0.05  # Small epsilon for evaluation
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Select action using epsilon-greedy policy
        action = get_epsilon_greedy_action(q_values, epsilon, self.action_size)
        
        return action
    
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
            float: Loss value
        """
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Update network if enough samples are available
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.memory.sample()
        
        # Preprocess batch
        states = self.preprocess_state(batch['states'])
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        next_states = self.preprocess_state(batch['next_states'])
        dones = torch.tensor(batch['dones'], dtype=torch.float32).to(self.device)
        
        # Compute Q-values for current states and actions
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network if needed
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, file_path=None):
        """
        Save agent's models and metrics.
        
        Args:
            file_path (str, optional): Path to save the model.
                                      If None, use default naming.
        """
        if file_path is None:
            file_path = os.path.join(self.model_dir, 'dqn_model.pth')
        
        # Save model, optimizer, and additional info
        additional_info = {
            'training_step': self.training_step,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay
        }
        
        save_model(self.q_network, self.optimizer, file_path, additional_info)
        
        # Save target network
        target_path = os.path.join(os.path.dirname(file_path), 'dqn_target_model.pth')
        torch.save(self.target_network.state_dict(), target_path)
        
        # Save accumulated metrics
        if hasattr(self, 'metrics_history'):
            metrics_path = os.path.join(os.path.dirname(file_path), 'dqn_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
        
        # Save additional metrics
        if hasattr(self, 'additional_metrics'):
            additional_metrics_path = os.path.join(os.path.dirname(file_path), 'dqn_additional_metrics.json')
            with open(additional_metrics_path, 'w') as f:
                json.dump(self.additional_metrics, f, indent=4)
    
    def load(self, file_path):
        """
        Load agent's models and metrics.
        
        Args:
            file_path (str): Path to the saved model
        """
        # Load model, optimizer, and additional info
        self.q_network, self.optimizer, additional_info = load_model(
            self.q_network,
            self.optimizer,
            file_path
        )
        
        # Update agent parameters
        self.training_step = additional_info.get('training_step', 0)
        self.epsilon_start = additional_info.get('epsilon_start', self.epsilon_start)
        self.epsilon_end = additional_info.get('epsilon_end', self.epsilon_end)
        self.epsilon_decay = additional_info.get('epsilon_decay', self.epsilon_decay)
        
        # Load target network
        target_path = os.path.join(os.path.dirname(file_path), 'dqn_target_model.pth')
        if os.path.exists(target_path):
            self.target_network.load_state_dict(torch.load(target_path))
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Load metrics
        metrics_path = os.path.join(os.path.dirname(file_path), 'dqn_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics_history = json.load(f)
        
        # Load additional metrics
        additional_metrics_path = os.path.join(os.path.dirname(file_path), 'dqn_additional_metrics.json')
        if os.path.exists(additional_metrics_path):
            with open(additional_metrics_path, 'r') as f:
                self.additional_metrics = json.load(f)
    
    # Add methods required by main.py
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform a single training step.
        
        Returns:
            float: Loss value (0.0 if no training was performed)
        """
        # Update network if enough samples are available
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.memory.sample()
        
        # Preprocess batch - Fix tensor shape issues
        # Convert states from [batch_size, height, width, channels] to [batch_size, channels, height, width]
        if isinstance(batch['states'], np.ndarray) and len(batch['states'].shape) == 4:
            # States are already batched, shape [batch_size, height, width, channels]
            states = torch.from_numpy(batch['states'].astype(np.float32) / 255.0)
            states = states.permute(0, 3, 1, 2).contiguous()  # Change to [batch_size, channels, height, width] and make contiguous
        else:
            states = self.preprocess_state(batch['states']).contiguous()
            
        # Do the same for next_states
        if isinstance(batch['next_states'], np.ndarray) and len(batch['next_states'].shape) == 4:
            next_states = torch.from_numpy(batch['next_states'].astype(np.float32) / 255.0)
            next_states = next_states.permute(0, 3, 1, 2).contiguous()
        else:
            next_states = self.preprocess_state(batch['next_states']).contiguous()
            
        # Convert batch to tensors
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32).to(self.device)
        
        # Normalize rewards for stability (helps with varying reward scales)
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8  # Add small epsilon to avoid division by zero
        normalized_rewards = (rewards - reward_mean) / reward_std
        
        # Move states to device
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        
        # Compute Q-values for current states and actions
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            # Use normalized rewards
            target_q_values = normalized_rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)  # Reduced from 10.0 to 1.0
        
        self.optimizer.step()
        
        # Update target network if needed
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
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
        
        # Get memory size - handle different possible attribute names
        if hasattr(self.memory, 'capacity'):
            memory_size = self.memory.capacity
        elif hasattr(self.memory, 'memory') and hasattr(self.memory.memory, 'maxlen'):
            memory_size = self.memory.memory.maxlen
        else:
            memory_size = len(self.memory)  # Fallback
        
        # Save hyperparameters
        results['hyperparameters'] = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'target_update': self.target_update,
            'batch_size': self.batch_size,
            'memory_size': memory_size
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def eval(self):
        """
        Set the agent to evaluation mode.
        """
        self.q_network.eval()
        self.target_network.eval() 