"""
Replay Buffer for off-policy RL algorithms
"""
import numpy as np
import random
from collections import deque, namedtuple

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    
    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        
        Args:
            state: current state observation
            action: action taken
            reward: reward received
            next_state: next state observation
            done: whether the episode has ended
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        
        Returns:
            Dictionary containing batch of experiences
        """
        experiences = random.sample(self.memory, k=min(self.batch_size, len(self.memory)))
        
        # Convert to numpy arrays
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer using importance sampling
    Based on "Prioritized Experience Replay" by Schaul et al.
    """
    
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta=0.4, beta_increment=1e-5):
        """
        Initialize a PrioritizedReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): determines how much prioritization is used (0 - no prioritization, 1 - full prioritization)
            beta (float): importance-sampling correction factor (0 - no correction, 1 - full correction)
            beta_increment (float): increment value for beta parameter over time
        """
        self.memory = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.position = 0
        self.size = 0
        
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory with max priority.
        
        Args:
            state: current state observation
            action: action taken
            reward: reward received
            next_state: next state observation
            done: whether the episode has ended
        """
        # Create experience tuple
        e = self.experience(state, action, reward, next_state, done)
        
        # Find max priority for new experience
        max_priority = max(self.priorities.max(), 1e-6)
        
        # Add to buffer
        if self.size < self.buffer_size:
            self.memory.append(e)
            self.size += 1
        else:
            self.memory[self.position] = e
        
        # Update priority
        self.priorities[self.position] = max_priority
        
        # Update position
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self):
        """
        Sample a batch of experiences from memory with prioritization.
        
        Returns:
            Dictionary containing batch of experiences and importance sampling weights
        """
        # Cannot sample if buffer is not filled yet
        if self.size < self.batch_size:
            # Return None or a smaller batch
            return None
        
        # Calculate sampling probabilities based on priorities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, self.batch_size, replace=False, p=probabilities)
        
        # Get experiences
        experiences = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to numpy arrays
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'weights': weights,
            'indices': indices
        }
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for sampled experiences based on TD error.
        
        Args:
            indices (list/array): Indices of the sampled experiences
            td_errors (list/array): TD errors for each sampled experience
        """
        for idx, td_error in zip(indices, td_errors):
            # Add small constant to prevent zero priority
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.size 