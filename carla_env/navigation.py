"""
Navigation utilities for the CARLA reinforcement learning environment.
This module provides tools for handling waypoints, generating paths, and tracking destinations.
"""
import math
import numpy as np
import carla
import random

class Waypoint:
    """A simple waypoint class to represent points in the world"""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z
        
    def distance_to(self, other):
        """Calculate distance to another waypoint or position tuple"""
        if isinstance(other, Waypoint):
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
        elif isinstance(other, tuple) and len(other) >= 2:
            return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)
        else:
            raise ValueError("Invalid position for distance calculation")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {"x": self.x, "y": self.y, "z": self.z}
    
    def to_carla_location(self):
        """Convert to CARLA Location object"""
        return carla.Location(x=self.x, y=self.y, z=self.z)
    
    @staticmethod
    def from_dict(data):
        """Create a waypoint from a dictionary"""
        return Waypoint(data["x"], data["y"], data.get("z", 0.0))
    
    @staticmethod
    def from_carla_location(location):
        """Create a waypoint from a CARLA Location object"""
        return Waypoint(location.x, location.y, location.z)
    
    def __str__(self):
        return f"Waypoint({self.x}, {self.y}, {self.z})"

class RouteManager:
    """Manages routes between waypoints and provides navigation guidance"""
    def __init__(self, world):
        self.world = world
        self.waypoints = []
        self.current_target_idx = 0
        self.target_threshold = 5.0  # Default distance threshold to consider target reached
        self.difficulty_levels = {
            'easy': {'min_dist': 20.0, 'max_dist': 40.0, 'num_points': 3},
            'medium': {'min_dist': 50.0, 'max_dist': 100.0, 'num_points': 5},
            'hard': {'min_dist': 100.0, 'max_dist': 200.0, 'num_points': 8}
        }
    
    def set_target_threshold(self, threshold):
        """Set the distance threshold for reaching a target"""
        self.target_threshold = threshold
    
    def get_current_target(self):
        """Get the current target waypoint"""
        if not self.waypoints or self.current_target_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.current_target_idx]
    
    def update_target(self, vehicle_location):
        """Update the current target based on vehicle position"""
        if not self.waypoints:
            return False
        
        # Check if we've reached the current target
        current_target = self.get_current_target()
        if current_target:
            loc = vehicle_location
            current_pos = (loc.x, loc.y, loc.z) if hasattr(loc, 'x') else (loc['x'], loc['y'], loc['z'])
            
            # Convert current_pos to tuple if it's a carla.Location
            if hasattr(current_pos, 'x'):
                current_pos = (current_pos.x, current_pos.y, current_pos.z)
                
            distance = math.sqrt((current_pos[0] - current_target.x)**2 + 
                                (current_pos[1] - current_target.y)**2)
            
            if distance < self.target_threshold:
                # Move to the next target
                self.current_target_idx += 1
                return True
        
        return False
    
    def is_route_complete(self):
        """Check if we've completed the entire route"""
        return self.current_target_idx >= len(self.waypoints)
    
    def get_distance_to_target(self, vehicle_location):
        """Calculate distance to the current target"""
        current_target = self.get_current_target()
        if not current_target:
            return float('inf')
            
        loc = vehicle_location
        current_pos = (loc.x, loc.y, loc.z) if hasattr(loc, 'x') else (loc['x'], loc['y'], loc['z'])
        
        # Convert current_pos to tuple if it's a carla.Location
        if hasattr(current_pos, 'x'):
            current_pos = (current_pos.x, current_pos.y, current_pos.z)
            
        return math.sqrt((current_pos[0] - current_target.x)**2 + 
                         (current_pos[1] - current_target.y)**2)
    
    def get_direction_to_target(self, vehicle_location, vehicle_rotation):
        """Calculate direction vector to current target relative to vehicle orientation"""
        current_target = self.get_current_target()
        if not current_target:
            return (0, 0)
            
        # Get vehicle location and forward vector
        yaw = math.radians(vehicle_rotation.yaw) if hasattr(vehicle_rotation, 'yaw') else math.radians(vehicle_rotation['yaw'])
        forward_vector = (math.cos(yaw), math.sin(yaw))
        
        # Get vector to target
        loc = vehicle_location
        current_pos = (loc.x, loc.y) if hasattr(loc, 'x') else (loc['x'], loc['y'])
        
        # Convert current_pos to tuple if it's a carla.Location
        if hasattr(current_pos, 'x'):
            current_pos = (current_pos.x, current_pos.y)
            
        target_vector = (current_target.x - current_pos[0], current_target.y - current_pos[1])
        
        # Normalize target vector
        target_distance = math.sqrt(target_vector[0]**2 + target_vector[1]**2)
        if target_distance > 0:
            target_vector = (target_vector[0] / target_distance, target_vector[1] / target_distance)
        
        # Calculate dot product and cross product to determine angle
        dot_product = forward_vector[0] * target_vector[0] + forward_vector[1] * target_vector[1]
        cross_product = forward_vector[0] * target_vector[1] - forward_vector[1] * target_vector[0]
        
        return (dot_product, cross_product)
    
    def generate_random_route(self, difficulty='easy', start_location=None):
        """Generate a random route with increasing difficulty"""
        self.waypoints = []
        self.current_target_idx = 0
        
        # Get difficulty parameters
        diff_params = self.difficulty_levels.get(difficulty, self.difficulty_levels['easy'])
        min_dist = diff_params['min_dist']
        max_dist = diff_params['max_dist']
        num_points = diff_params['num_points']
        
        # Get road waypoints near the spawn location
        if start_location is None:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                return []
            start_location = random.choice(spawn_points).location
        
        # Generate waypoints along roads
        current_pos = start_location
        waypoint = self.world.get_map().get_waypoint(current_pos)
        
        for i in range(num_points):
            # Move forward along the road
            distance = random.uniform(min_dist, max_dist)
            next_waypoints = waypoint.next(distance)
            
            if not next_waypoints:
                break
                
            # If at a junction, randomly choose a path
            if waypoint.is_junction:
                waypoint = random.choice(next_waypoints)
            else:
                waypoint = next_waypoints[0]
            
            # Add to our waypoint list
            self.waypoints.append(Waypoint.from_carla_location(waypoint.transform.location))
            
        return self.waypoints
    
    def draw_waypoints(self, debug_helper, life_time=10.0):
        """Visualize the waypoints for debugging"""
        if not self.waypoints:
            return
            
        # Define colors based on status
        color_current = carla.Color(0, 255, 0)   # Green for current target
        color_future = carla.Color(0, 0, 255)    # Blue for future targets
        color_reached = carla.Color(255, 0, 0)   # Red for reached targets
        
        for i, waypoint in enumerate(self.waypoints):
            # Determine color based on target status
            if i < self.current_target_idx:
                color = color_reached
            elif i == self.current_target_idx:
                color = color_current
            else:
                color = color_future
                
            # Draw a sphere at the waypoint
            location = waypoint.to_carla_location()
            debug_helper.draw_point(location, size=0.2, color=color, life_time=life_time)
            
            # Draw text showing the waypoint index
            text_location = carla.Location(x=location.x, y=location.y, z=location.z + 1.0)
            debug_helper.draw_string(text_location, str(i), draw_shadow=False, 
                                   color=carla.Color(255, 255, 255), life_time=life_time) 