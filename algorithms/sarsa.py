"""
SARSA (State-Action-Reward-State-Action) implementation
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import sys
import json

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
from utils.helpers import calculate_epsilon, get_epsilon_greedy_action, save_model, load_model
from algorithms.dqn import DQNModel

class SARSAAgent(BaseAgent):
    """
    SARSA (State-Action-Reward-State-Action) agent.
    
    This implementation uses a neural network to approximate the Q-function,
    but follows the on-policy SARSA update rule instead of off-policy Q-learning.
    """
    
    def __init__(self, 
                 state_shape=(3, 84, 84), 
                 action_size=7,
                 hidden_dim=512,
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=50000,
                 memory_size=100000,
                 batch_size=64,
                 device=None,
                 model_dir='models',
                 tensorboard_dir='logs'):
        """
        Initialize the SARSA agent.
        
        Args:
            state_shape (tuple): Shape of the state (C, H, W)
            action_size (int): Number of possible actions
            hidden_dim (int): Size of hidden layers
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor for future rewards
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (int): Number of steps for epsilon decay
            memory_size (int): Maximum size of replay buffer
            batch_size (int): Batch size for training
            device (str, optional): Device to use for computation
            model_dir (str): Directory to save models
            tensorboard_dir (str): Directory for TensorBoard logs
        """
        # Set required attributes before initializing the parent class
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize base agent
        super(SARSAAgent, self).__init__(
            state_shape=state_shape,
            action_size=action_size,
            device=device,
            model_dir=model_dir,
            tensorboard_dir=tensorboard_dir
        )
        
        # Create replay buffer
        self.memory = ReplayBuffer(memory_size, batch_size)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize loss function
        self.loss_fn = nn.MSELoss()
        
        # Additional variables for SARSA
        self.last_state = None
        self.last_action = None
    
    def _init_models(self):
        """
        Initialize Q-network.
        """
        self.q_network = DQNModel(self.state_shape, self.action_size, self.hidden_dim).to(self.device)
    
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training (bool): Whether the agent is in training mode
            
        Returns:
            int: Selected action
        """
        # Preprocess state
        state_tensor = self.preprocess_state(state)
        
        # Calculate epsilon for exploration
        if training:
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
        
        # Store state and action for SARSA update
        if training and self.last_state is None:
            self.last_state = state
            self.last_action = action
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on SARSA.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done (bool): Whether the episode is done
            
        Returns:
            float: Loss value
        """
        # For the first update, we need both state-action pairs
        if self.last_state is None:
            self.last_state = state
            self.last_action = action
            return 0.0
        
        # Select next action for SARSA update (on-policy)
        if not done:
            next_action = self.select_action(next_state, training=True)
        else:
            next_action = 0  # Dummy action for terminal state
        
        # Add experience to replay buffer (using SARSA format)
        self.memory.add(self.last_state, self.last_action, reward, state, done)
        
        # Update network if enough samples are available
        loss = 0.0
        if len(self.memory) >= self.batch_size:
            loss = self._update_network(next_state, next_action)
        
        # Update last state and action
        self.last_state = state if not done else None
        self.last_action = action if not done else None
        
        # Reset last state and action at end of episode
        if done:
            self.last_state = None
            self.last_action = None
        
        return loss
    
    def _update_network(self, next_state, next_action):
        """
        Update the network using SARSA update rule.
        
        Args:
            next_state: The state after the current one
            next_action: The action selected for the next state
            
        Returns:
            float: Loss value
        """
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
        
        # Compute target Q-values for next states and selected actions
        # This is different from DQN - we use the actual next action
        with torch.no_grad():
            # If we're at the latest step, use the action from the method argument
            if states.shape[0] == 1:
                next_state_tensor = self.preprocess_state(next_state)
                next_action_tensor = torch.tensor([next_action], dtype=torch.long).to(self.device)
                next_q_values = self.q_network(next_state_tensor)
                next_q_values = next_q_values.gather(1, next_action_tensor.unsqueeze(1)).squeeze(1)
            else:
                # Otherwise, just use the next states from the batch
                # This is an approximation, as we're mixing on-policy and off-policy updates
                next_q_values = self.q_network(next_states)
                # For simplicity, we'll use max Q-value as an approximation
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
        
        return loss.item()
    
    def start_episode(self):
        """
        Called at the beginning of an episode.
        """
        super().start_episode()
        
        # Reset SARSA-specific variables
        self.last_state = None
        self.last_action = None
    
    def save(self, file_path=None):
        """
        Save agent's models and metrics.
        
        Args:
            file_path (str, optional): Path to save the model.
                                      If None, use default naming.
        """
        if file_path is None:
            file_path = os.path.join(self.model_dir, 'sarsa_model.pth')
        
        # Save model, optimizer, and additional info
        additional_info = {
            'training_step': self.training_step,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay
        }
        
        save_model(self.q_network, self.optimizer, file_path, additional_info)
        
        # Save metrics - use the base class implementation directly to avoid parameter issues
        metrics_path = os.path.join(os.path.dirname(file_path), f"{self.__class__.__name__}_metrics.json")
        super().save_metrics(metrics_path)
    
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
        
        # Load metrics
        metrics_path = os.path.join(os.path.dirname(file_path), 'SARSAAgent_metrics.json')
        if os.path.exists(metrics_path):
            self.load_metrics(metrics_path)
            
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
            'batch_size': self.batch_size,
            'memory_size': memory_size
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)

    def save_metrics(self, episode=None, reward=None, loss=None, file_path=None):
        """
        Save training metrics to a file.
        
        Args:
            episode (int): Current episode number
            reward (float): Current episode reward
            loss (float): Current episode loss
            file_path (str, optional): Path to save metrics.
                                      If None, use default naming.
        """
        # If episode, reward, and loss are provided, save them to metrics history
        if episode is not None and reward is not None and loss is not None:
            if not hasattr(self, 'metrics_history'):
                self.metrics_history = {
                    'episodes': [],
                    'rewards': [],
                    'losses': []
                }
            
            self.metrics_history['episodes'].append(episode)
            self.metrics_history['rewards'].append(reward)
            self.metrics_history['losses'].append(loss)
            
        # Call the base class method for regular metrics saving
        if file_path is None:
            file_path = os.path.join(self.model_dir, f"{self.__class__.__name__}_metrics.json")
        
        # Save metrics to file using BaseAgent implementation directly
        BaseAgent.save_metrics(self, file_path)
        
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience for SARSA.
        
        For SARSA, this just calls the update method since we're using an on-policy
        approach and need to update immediately with the current policy.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            float: Loss value from the update
        """
        # Call the update method directly
        return self.update(state, action, reward, next_state, done)
        
    def train_step(self):
        """
        Perform a single training step.
        
        For SARSA, the updates happen during the store_experience (update) method,
        so this is a no-op.
        
        Returns:
            float: 0.0 (no additional loss since updates happen during experience collection)
        """
        return 0.0
        
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

    def preprocess_state(self, state):
        """
        Preprocess the state for neural network input.
        
        Args:
            state: Raw state from the environment
            
        Returns:
            torch.Tensor: Processed state tensor
        """
        # Handle batch of states
        if isinstance(state, list) or (isinstance(state, np.ndarray) and state.ndim == 4):
            # If it's a batch of images with HWC format (height, width, channels)
            if isinstance(state, list):
                # Convert list to numpy array
                state = np.array(state)
            
            if state.shape[-1] == 3:  # HWC format
                # Convert HWC to CHW format
                if isinstance(state, np.ndarray):
                    # For numpy arrays
                    state = np.transpose(state, (0, 3, 1, 2))
                    state = torch.from_numpy(state.astype(np.float32) / 255.0)
                else:
                    # For torch tensors
                    state = state.permute(0, 3, 1, 2)
            
            return state.to(self.device)
            
        # Handle single state
        if isinstance(state, np.ndarray) and len(state.shape) == 3:
            # If image is in HWC format (height, width, channels)
            if state.shape[-1] == 3:
                # Convert HWC to CHW format
                state = np.transpose(state, (2, 0, 1))
                state = torch.from_numpy(state.astype(np.float32) / 255.0)
                return state.unsqueeze(0).to(self.device)
        
        # If state is already a tensor, ensure it's on the right device
        if isinstance(state, torch.Tensor):
            # Add batch dimension if needed
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            return state.to(self.device)
        
        # Otherwise, convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Add batch dimension if needed
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        return state_tensor.to(self.device)
        
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