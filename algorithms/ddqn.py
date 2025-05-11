"""
Double Deep Q-Network (DDQN) implementation
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
from algorithms.dqn import DQNAgent, DQNModel
from utils.helpers import save_model, load_model

class DDQNAgent(DQNAgent):
    """
    Double Deep Q-Network (DDQN) agent.
    
    This implementation extends the DQN algorithm to reduce overestimation
    of Q-values by using separate networks for action selection and evaluation.
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
        Initialize the DDQN agent.
        
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
        # Initialize DQN agent
        super(DDQNAgent, self).__init__(
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
        
        # Normalize rewards for stability
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8  # Add small epsilon to avoid division by zero
        normalized_rewards = (rewards - reward_mean) / reward_std
        
        # Compute Q-values for current states and actions
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Use online network to select actions
            online_next_q_values = self.q_network(next_states)
            online_next_actions = online_next_q_values.argmax(1, keepdim=True)
            
            # Use target network to evaluate actions
            target_next_q_values = self.target_network(next_states)
            target_next_q_values = target_next_q_values.gather(1, online_next_actions).squeeze(1)
            
            # Compute target Q-values using normalized rewards
            target_q_values = normalized_rewards + (1 - dones) * self.gamma * target_next_q_values
        
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
    
    def save(self, file_path=None):
        """
        Save agent's models and metrics.
        
        Args:
            file_path (str, optional): Path to save the model.
                                      If None, use default naming.
        """
        if file_path is None:
            file_path = os.path.join(self.model_dir, 'ddqn_model.pth')
        
        # Save model, optimizer, and additional info
        additional_info = {
            'training_step': self.training_step,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay
        }
        
        save_model(self.q_network, self.optimizer, file_path, additional_info)
        
        # Save target network
        target_path = os.path.join(os.path.dirname(file_path), 'ddqn_target_model.pth')
        torch.save(self.target_network.state_dict(), target_path)
        
        # Save metrics - use the base class implementation directly to avoid parameter issues
        metrics_path = os.path.join(os.path.dirname(file_path), f"{self.__class__.__name__}_metrics.json")
        super(DQNAgent, self).save_metrics(metrics_path)
    
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
        target_path = os.path.join(os.path.dirname(file_path), 'ddqn_target_model.pth')
        if os.path.exists(target_path):
            self.target_network.load_state_dict(torch.load(target_path))
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Load metrics
        metrics_path = os.path.join(os.path.dirname(file_path), 'DDQNAgent_metrics.json')
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
            
        # Call the parent class method for regular metrics saving
        if file_path is None:
            file_path = os.path.join(self.model_dir, f"{self.__class__.__name__}_metrics.json")
        
        # Save metrics to file using parent implementation directly
        super(DQNAgent, self).save_metrics(file_path) 