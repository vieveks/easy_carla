"""
Reinforcement Learning environment interface for CARLA
"""
import gym
from gym import spaces
import numpy as np
import time
import sys
import os
import cv2
import random
import math
from collections import deque

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from carla_env.carla_client import CarlaClient

class CarlaEnv(gym.Env):
    """
    Custom Environment for CARLA that follows the gym interface.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, host=config.CARLA_HOST, port=config.CARLA_PORT, 
                 timeout=config.CARLA_TIMEOUT, town=config.CARLA_MAP,
                 image_shape=(84, 84, 3)):
        super(CarlaEnv, self).__init__()
        
        # Initialize CARLA client
        self.client = CarlaClient(host, port, timeout, town)
        self.connected = False
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(config.DISCRETE_ACTIONS))
        
        # Image-based observation space
        self.image_shape = image_shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.image_shape, dtype=np.uint8
        )
        
        # Episode settings
        self.max_episode_steps = config.MAX_EPISODE_STEPS
        self.current_step = 0
        self.total_reward = 0.0
        self.previous_location = None
        self.previous_waypoint = None
        self.previous_action = None
        self.episode_statistics = {
            'rewards': [],
            'centerline_deviations': [],
            'junction_actions': {
                'left': 0,
                'right': 0,
                'forward': 0,
                'other': 0
            }
        }
        
        # Movement tracking
        self.last_locations = deque(maxlen=10)
        
        # Performance metrics
        self.centerline_deviations = []
        self.collision_count = 0
        self.lane_invasion_count = 0
        self.target_reached = False
        
    def connect(self):
        """Connect to the CARLA server"""
        if not self.connected:
            if self.client.connect():
                self.connected = True
                return True
        
        return self.connected
    
    def reset(self, seed=None):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        # Connect to CARLA if not already connected
        if not self.connected:
            if not self.connect():
                raise Exception("Could not connect to CARLA server")
        
        # Set the seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset client (spawn vehicle, setup sensors)
        self.client.reset()
        
        # Reset environment variables
        self.current_step = 0
        self.total_reward = 0.0
        self.previous_location = None
        self.previous_waypoint = None
        self.previous_action = None
        self.last_locations.clear()
        self.centerline_deviations = []
        
        # Track current vehicle position
        vehicle_state = self.client.get_vehicle_state()
        if vehicle_state:
            loc = vehicle_state['location']
            self.previous_location = (loc['x'], loc['y'], loc['z'])
            self.last_locations.append(self.previous_location)
            
            # Store initial waypoint
            distance = self.client.get_distance_from_center()
            self.previous_waypoint = distance
        
        # Tick the simulation to get initial observation
        self.client.tick()
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        """
        self.current_step += 1
        self.previous_action = action
        
        # Apply action to the vehicle
        self.client.apply_action(action)
        
        # Advance simulation
        self.client.tick()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward, reward_info = self._compute_reward()
        self.total_reward += reward
        
        # Check termination conditions
        done = self._is_done()
        
        # Get additional info
        info = self._get_info()
        info.update(reward_info)
        
        # Record metrics
        if not done:
            # Record centerline deviation
            centerline_deviation = self.client.get_distance_from_center()
            self.centerline_deviations.append(centerline_deviation)
            
            # Record junction actions
            if self.client.is_at_junction():
                if action == 2:  # Left
                    self.episode_statistics['junction_actions']['left'] += 1
                elif action == 3:  # Right
                    self.episode_statistics['junction_actions']['right'] += 1
                elif action == 1:  # Forward
                    self.episode_statistics['junction_actions']['forward'] += 1
                else:
                    self.episode_statistics['junction_actions']['other'] += 1
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """
        Get the current observation (camera image)
        """
        # Get camera image
        image = self.client.get_observation()
        
        # If no image is available, return a blank image
        if image is None:
            return np.zeros(self.image_shape, dtype=np.uint8)
        
        # Resize image if necessary
        if image.shape != self.image_shape:
            image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))
        
        return image
    
    def _compute_reward(self):
        """
        Calculate the reward for the current state.
        """
        reward = 0.0
        info = {}
        
        # Retrieve current vehicle state
        vehicle_state = self.client.get_vehicle_state()
        
        # Collision penalty
        if self.client.has_collided():
            reward += config.REWARD_COLLISION
            info['collision'] = True
            self.collision_count += 1
        else:
            info['collision'] = False
        
        # Lane invasion penalty
        if self.client.has_crossed_lane():
            reward += config.REWARD_LANE_INVASION
            info['lane_invasion'] = True
            self.lane_invasion_count += 1
        else:
            info['lane_invasion'] = False
        
        # Reset sensor flags for next step
        self.client.reset_sensors()
        
        if vehicle_state:
            # Current position
            loc = vehicle_state['location']
            current_location = (loc['x'], loc['y'], loc['z'])
            self.last_locations.append(current_location)
            
            # Speed reward
            speed = vehicle_state['speed']  # km/h
            info['speed'] = speed
            
            # Forward movement reward
            if self.previous_location:
                # Calculate distance moved in the forward direction
                move_reward = config.REWARD_FORWARD
                reward += move_reward
                info['move_reward'] = move_reward
            
            # Centerline distance penalty
            distance_to_center = self.client.get_distance_from_center()
            centerline_penalty = config.REWARD_CENTERLINE_DISTANCE * distance_to_center
            reward += centerline_penalty
            info['centerline_penalty'] = centerline_penalty
            info['distance_to_center'] = distance_to_center
            
            # Speed reward (prefer moving at a moderate speed)
            speed_reward = config.REWARD_SPEED * min(speed, 30) / 30.0  # Cap at 30 km/h
            reward += speed_reward
            info['speed_reward'] = speed_reward
            
            # Update previous location
            self.previous_location = current_location
            
            # Check if vehicle is stuck (not moving for several steps)
            if len(self.last_locations) >= 10:
                # Calculate average movement over last 10 steps
                first_loc = self.last_locations[0]
                last_loc = self.last_locations[-1]
                dist = math.sqrt(
                    (last_loc[0] - first_loc[0])**2 + 
                    (last_loc[1] - first_loc[1])**2
                )
                
                # If vehicle hasn't moved much
                if dist < 1.0:  # Less than 1 meter in 10 steps
                    reward += -1.0  # Small penalty for being stuck
                    info['stuck'] = True
                else:
                    info['stuck'] = False
        
        # Time penalty (small penalty for each step)
        reward += config.REWARD_TIME_PENALTY
        info['time_penalty'] = config.REWARD_TIME_PENALTY
        
        # Total reward for this step
        info['reward'] = reward
        
        return reward, info
    
    def _is_done(self):
        """
        Check if the episode should end.
        """
        # Episode ends on collision
        if self.client.has_collided():
            return True
        
        # Check if target reached (not implemented yet)
        if self.target_reached:
            return True
        
        # Check if vehicle has stopped moving for too long
        if len(self.last_locations) >= 10:
            # Calculate average movement over last 10 steps
            first_loc = self.last_locations[0]
            last_loc = self.last_locations[-1]
            dist = math.sqrt(
                (last_loc[0] - first_loc[0])**2 + 
                (last_loc[1] - first_loc[1])**2
            )
            
            # If vehicle hasn't moved much for a long time
            if dist < 0.5 and self.current_step > 50:  # Less than 0.5 meters and more than 50 steps
                return True
        
        return False
    
    def _get_info(self):
        """
        Return additional information for debugging and monitoring.
        """
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
        }
        
        # Add vehicle state if available
        vehicle_state = self.client.get_vehicle_state()
        if vehicle_state:
            info['vehicle_state'] = vehicle_state
            info['distance_to_center'] = self.client.get_distance_from_center()
            info['at_junction'] = self.client.is_at_junction()
        
        return info
    
    def render(self, mode='human'):
        """
        Render the environment. In this case, CARLA already visualizes the simulation.
        """
        # CARLA handles visualization itself
        pass
    
    def close(self):
        """
        Clean up resources when environment is no longer needed.
        """
        if self.connected:
            self.client.disconnect()
            self.connected = False
    
    def get_episode_statistics(self):
        """
        Get statistics for the current episode.
        """
        # Calculate average centerline deviation
        avg_centerline_deviation = 0.0
        if self.centerline_deviations:
            avg_centerline_deviation = sum(self.centerline_deviations) / len(self.centerline_deviations)
            
        # Prepare statistics dictionary
        stats = {
            'total_reward': self.total_reward,
            'steps': self.current_step,
            'collision_count': self.collision_count,
            'lane_invasion_count': self.lane_invasion_count,
            'avg_centerline_deviation': avg_centerline_deviation,
            'junction_actions': self.episode_statistics['junction_actions'],
        }
        
        return stats 