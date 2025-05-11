"""
Dueling Deep Q-Network implementation
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import json

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.ddqn import DDQNAgent
from utils.helpers import save_model, load_model

class DuelingDQNModel(nn.Module):
    """
    Neural network model for Dueling DQN.
    
    This network separates state value and action advantage estimation
    to enable better policy evaluation.
    """
    
    def __init__(self, state_shape, action_size, hidden_dim=512):
        """
        Initialize the model.
        
        Args:
            state_shape (tuple): Shape of the state (C, H, W)
            action_size (int): Number of possible actions
            hidden_dim (int): Size of hidden layers
        """
        super(DuelingDQNModel, self).__init__()
        
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
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
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
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class DuelingDQNAgent(DDQNAgent):
    """
    Dueling Deep Q-Network (Dueling DQN) agent.
    
    This implementation extends Double DQN by using a dueling architecture
    to separately estimate state value and action advantages.
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
        Initialize the Dueling DQN agent.
        
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
        # Initialize the base class without initializing models
        # We'll override _init_models to use the dueling architecture
        self.state_shape = state_shape
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        
        # Call parent's init after setting required attributes
        super(DuelingDQNAgent, self).__init__(
            state_shape=state_shape,
            action_size=action_size,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update=target_update,
            memory_size=memory_size,
            batch_size=batch_size,
            device=device,
            model_dir=model_dir,
            tensorboard_dir=tensorboard_dir
        )
    
    def _init_models(self):
        """
        Initialize Dueling DQN networks.
        """
        self.q_network = DuelingDQNModel(self.state_shape, self.action_size, self.hidden_dim).to(self.device)
        self.target_network = DuelingDQNModel(self.state_shape, self.action_size, self.hidden_dim).to(self.device)
        
        # Initialize target network with same weights as Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
    
    def save(self, file_path=None):
        """
        Save agent's models and metrics.
        
        Args:
            file_path (str, optional): Path to save the model.
                                      If None, use default naming.
        """
        if file_path is None:
            file_path = os.path.join(self.model_dir, 'dueling_dqn_model.pth')
        
        # Save model, optimizer, and additional info
        additional_info = {
            'training_step': self.training_step,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay
        }
        
        save_model(self.q_network, self.optimizer, file_path, additional_info)
        
        # Save target network
        target_path = os.path.join(os.path.dirname(file_path), 'dueling_dqn_target_model.pth')
        torch.save(self.target_network.state_dict(), target_path)
        
        # Save metrics - use the base class implementation directly to avoid parameter issues
        metrics_path = os.path.join(os.path.dirname(file_path), f"{self.__class__.__name__}_metrics.json")
        super(DDQNAgent, self).save_metrics(metrics_path)
    
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
        target_path = os.path.join(os.path.dirname(file_path), 'dueling_dqn_target_model.pth')
        if os.path.exists(target_path):
            self.target_network.load_state_dict(torch.load(target_path))
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Load metrics
        metrics_path = os.path.join(os.path.dirname(file_path), 'DuelingDQNAgent_metrics.json')
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
            'target_update': self.target_update,
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
        
        # Save metrics to file using parent implementation directly
        super(DDQNAgent, self).save_metrics(file_path) 