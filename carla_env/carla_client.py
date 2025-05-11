"""
Module to handle CARLA client connection and actor management
"""
import random
import time
import numpy as np
import carla
import pygame
import sys
import os
import cv2
import math
from collections import deque

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from carla_env.navigation import RouteManager, Waypoint

class CarlaSensor:
    """Base class for CARLA sensors"""
    def __init__(self, world, vehicle, sensor_type, transform, sensor_options={}):
        self.world = world
        self.vehicle = vehicle
        self.sensor = None
        self.sensor_type = sensor_type
        self.transform = transform
        self.sensor_options = sensor_options
        self.data = None
        self.callback = None

    def setup_sensor(self):
        """Create and attach the sensor to the vehicle"""
        blueprint = self.world.get_blueprint_library().find(self.sensor_type)
        
        for key, value in self.sensor_options.items():
            blueprint.set_attribute(key, str(value))
            
        self.sensor = self.world.spawn_actor(
            blueprint,
            self.transform,
            attach_to=self.vehicle)
        
        # Set up data reception
        self.sensor.listen(self._sensor_callback)
    
    def _sensor_callback(self, data):
        """Process sensor data when received"""
        self.data = data
        if self.callback:
            self.callback(data)
            
    def destroy(self):
        """Clean up the sensor"""
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

class CameraRGB(CarlaSensor):
    """RGB camera sensor implementation"""
    def __init__(self, world, vehicle, width=84, height=84, fov=90, 
                 pos_x=1.5, pos_y=0.0, pos_z=2.4, 
                 pitch=0, roll=0, yaw=0):
        
        transform = carla.Transform(
            carla.Location(x=pos_x, y=pos_y, z=pos_z),
            carla.Rotation(pitch=pitch, roll=roll, yaw=yaw)
        )
        
        sensor_options = {
            'image_size_x': width,
            'image_size_y': height,
            'fov': fov
        }
        
        super().__init__(world, vehicle, 'sensor.camera.rgb', transform, sensor_options)
        self.image_array = np.zeros((height, width, 3), dtype=np.uint8)
        
    def _sensor_callback(self, image):
        """Process camera image"""
        super()._sensor_callback(image)
        
        # Convert image data to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Drop alpha channel
        
        self.image_array = array

class CollisionSensor(CarlaSensor):
    """Collision sensor implementation"""
    def __init__(self, world, vehicle):
        transform = carla.Transform(carla.Location(0, 0, 0))
        super().__init__(world, vehicle, 'sensor.other.collision', transform)
        self.collision_history = []
        self.has_collided = False
        
    def _sensor_callback(self, event):
        """Process collision event"""
        super()._sensor_callback(event)
        
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        self.collision_history.append((event.frame, intensity))
        self.has_collided = True
        
    def reset(self):
        """Reset collision history"""
        self.collision_history = []
        self.has_collided = False

class LaneInvasionSensor(CarlaSensor):
    """Lane invasion sensor implementation"""
    def __init__(self, world, vehicle):
        transform = carla.Transform(carla.Location(0, 0, 0))
        super().__init__(world, vehicle, 'sensor.other.lane_invasion', transform)
        self.invasion_history = []
        self.crossed_lane = False
        
    def _sensor_callback(self, event):
        """Process lane invasion event"""
        super()._sensor_callback(event)
        
        self.invasion_history.append((event.frame, event.crossed_lane_markings))
        self.crossed_lane = True
        
    def reset(self):
        """Reset lane invasion history"""
        self.invasion_history = []
        self.crossed_lane = False

class CarlaClient:
    """Main class for handling CARLA client connection and actors"""
    def __init__(self, 
                 host=config.CARLA_HOST, 
                 port=config.CARLA_PORT, 
                 timeout=config.CARLA_TIMEOUT,
                 town=config.CARLA_MAP):
        
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.spectator = None
        self.sensors = {}
        
        self.host = host
        self.port = port
        self.timeout = timeout
        self.town = town
        
        # Navigation attributes
        self.route_manager = None
        self.navigation_enabled = config.NAVIGATION_ENABLED
        self.route_visualization = config.ROUTE_VISUALIZATION
        self.current_difficulty = config.ROUTE_DIFFICULTY
        
        # Initialize pygame for visualization if needed
        pygame.init()
        
    def connect(self):
        """Connect to the CARLA server"""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            
            # Load desired map
            self.world = self.client.load_world(self.town)
            self.blueprint_library = self.world.get_blueprint_library()
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = config.FIXED_DELTA_SECONDS
            self.world.apply_settings(settings)
            
            # Set weather
            weather = getattr(carla.WeatherParameters, config.WEATHER)
            self.world.set_weather(weather)
            
            print(f"Connected to CARLA server at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"Error connecting to CARLA server: {e}")
            return False
    
    def spawn_vehicle(self, vehicle_type=config.VEHICLE_TYPE, spawn_point_idx=config.SPAWN_POINT_INDEX):
        """Spawn the ego vehicle at a specified spawn point"""
        if not self.world:
            print("Not connected to CARLA server")
            return False
        
        # Get available spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        
        if not spawn_points:
            print("No spawn points available")
            return False
        
        # Select spawn point
        if spawn_point_idx < len(spawn_points):
            spawn_point = spawn_points[spawn_point_idx]
        else:
            spawn_point = random.choice(spawn_points)
        
        # Find vehicle blueprint
        blueprint = self.blueprint_library.find(vehicle_type)
        
        # Set color
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
            
        # Set role name for hero vehicle
        blueprint.set_attribute('role_name', 'hero')
        
        # Try to spawn the vehicle
        self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        
        if not self.vehicle:
            print("Failed to spawn vehicle")
            return False
        
        # Wait for the world to be ready
        self.world.tick()
        
        # Set up spectator to follow the vehicle
        self.spectator = self.world.get_spectator()
        transform = carla.Transform(self.vehicle.get_transform().location + carla.Location(z=3),
                                  carla.Rotation(pitch=-30))
        self.spectator.set_transform(transform)
        
        # Initialize the route manager if navigation is enabled
        if self.navigation_enabled:
            self.route_manager = RouteManager(self.world)
            self.route_manager.set_target_threshold(config.WAYPOINT_THRESHOLD)
            self.route_manager.generate_random_route(difficulty=self.current_difficulty, 
                                                  start_location=spawn_point.location)
            
            # Visualize the route if enabled
            if self.route_visualization:
                self.visualize_route()
                
        print(f"Spawned {vehicle_type}")
        return True
    
    def setup_sensors(self):
        """Set up all sensors defined in config"""
        if not self.vehicle:
            print("Vehicle not spawned yet")
            return False
        
        # Setup RGB camera
        camera_config = config.SENSORS.get('rgb_camera', {})
        camera = CameraRGB(
            self.world, 
            self.vehicle,
            width=camera_config.get('image_size_x', 84),
            height=camera_config.get('image_size_y', 84),
            fov=camera_config.get('fov', 90),
            pos_x=camera_config.get('position_x', 1.5),
            pos_y=camera_config.get('position_y', 0),
            pos_z=camera_config.get('position_z', 2.4),
            pitch=camera_config.get('rotation_pitch', 0),
            roll=camera_config.get('rotation_roll', 0),
            yaw=camera_config.get('rotation_yaw', 0)
        )
        camera.setup_sensor()
        self.sensors['camera'] = camera
        
        # Setup collision sensor
        collision = CollisionSensor(self.world, self.vehicle)
        collision.setup_sensor()
        self.sensors['collision'] = collision
        
        # Setup lane invasion sensor
        lane_invasion = LaneInvasionSensor(self.world, self.vehicle)
        lane_invasion.setup_sensor()
        self.sensors['lane_invasion'] = lane_invasion
        
        print("All sensors have been set up")
        return True
    
    def tick(self):
        """Advance the simulation by one step"""
        if not self.world:
            return
            
        # Advance simulation
        self.world.tick()
        
        # Update spectator to follow the vehicle
        if self.vehicle and self.spectator:
            vehicle_transform = self.vehicle.get_transform()
            camera_transform = carla.Transform(
                vehicle_transform.location + carla.Location(z=3, x=-5),
                carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw)
            )
            self.spectator.set_transform(camera_transform)
        
        return True
    
    def apply_control(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        """Apply control to the vehicle"""
        if self.vehicle:
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=hand_brake,
                reverse=reverse
            )
            self.vehicle.apply_control(control)
            return True
        return False
    
    def apply_action(self, action_idx):
        """Apply action from discrete action space"""
        if action_idx in config.DISCRETE_ACTIONS:
            throttle_brake, steer = config.DISCRETE_ACTIONS[action_idx]
            
            # Apply throttle or brake based on sign
            if throttle_brake >= 0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake
                
            return self.apply_control(throttle=throttle, steer=steer, brake=brake)
        
        return False
    
    def get_observation(self):
        """Get the current observation (camera image)"""
        if 'camera' in self.sensors:
            return self.sensors['camera'].image_array
        return None
    
    def get_vehicle_state(self):
        """Get the current vehicle state (position, velocity, etc.)"""
        if self.vehicle:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            control = self.vehicle.get_control()
            
            # Calculate speed in km/h
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            return {
                'location': {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z
                },
                'rotation': {
                    'pitch': transform.rotation.pitch,
                    'roll': transform.rotation.roll,
                    'yaw': transform.rotation.yaw
                },
                'velocity': {
                    'x': velocity.x,
                    'y': velocity.y,
                    'z': velocity.z
                },
                'speed': speed,
                'control': {
                    'throttle': control.throttle,
                    'steer': control.steer,
                    'brake': control.brake
                }
            }
        
        return None
    
    def has_collided(self):
        """Check if the vehicle has collided"""
        if 'collision' in self.sensors:
            return self.sensors['collision'].has_collided
        return False
    
    def has_crossed_lane(self):
        """Check if the vehicle has crossed a lane marking"""
        if 'lane_invasion' in self.sensors:
            return self.sensors['lane_invasion'].crossed_lane
        return False
    
    def get_distance_from_center(self):
        """Calculate the distance from the center line of the current lane"""
        if self.vehicle:
            # Get the waypoint for the current vehicle location
            waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
            
            # Calculate distance to waypoint (center of lane)
            vehicle_loc = self.vehicle.get_location()
            waypoint_loc = waypoint.transform.location
            
            # Euclidean distance in 2D (ignoring height)
            distance = math.sqrt((vehicle_loc.x - waypoint_loc.x)**2 + 
                                 (vehicle_loc.y - waypoint_loc.y)**2)
            
            return distance
        
        return 0.0
    
    def reset_sensors(self):
        """Reset all sensors"""
        if 'collision' in self.sensors:
            self.sensors['collision'].reset()
            
        if 'lane_invasion' in self.sensors:
            self.sensors['lane_invasion'].reset()
    
    def reset(self, spawn_point_idx=None):
        """Reset the vehicle and sensors"""
        if spawn_point_idx is None:
            spawn_point_idx = config.SPAWN_POINT_INDEX
            
        # Destroy current vehicle and sensors
        self.destroy()
        
        # Spawn new vehicle
        self.spawn_vehicle(spawn_point_idx=spawn_point_idx)
        
        # Setup sensors
        self.setup_sensors()
        
        # Tick once to initialize everything
        self.tick()
        
        return True
    
    def destroy(self):
        """Clean up all actors"""
        # Destroy sensors
        for sensor in self.sensors.values():
            sensor.destroy()
        self.sensors = {}
        
        # Destroy vehicle
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
    
    def disconnect(self):
        """Clean up and disconnect from CARLA server"""
        self.destroy()
        
        # Reset world settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        
        # Cleanup pygame
        pygame.quit()
        
        print("Disconnected from CARLA server")
        
    def is_at_junction(self):
        """Check if the vehicle is at a junction"""
        if self.vehicle:
            # Get the waypoint for the current vehicle location
            waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
            return waypoint.is_junction
        return False
    
    def update_navigation(self):
        """Update navigation and target waypoints based on vehicle position"""
        if not self.navigation_enabled or not self.route_manager or not self.vehicle:
            return False
            
        # Update the current target based on vehicle position
        vehicle_loc = self.vehicle.get_location()
        target_updated = self.route_manager.update_target(vehicle_loc)
        
        # Check if we completed the route
        route_complete = self.route_manager.is_route_complete()
        
        # Update route visualization if needed
        if target_updated and self.route_visualization:
            self.visualize_route()
            
        return route_complete
    
    def get_navigation_info(self):
        """Get navigation information for the current state"""
        if not self.navigation_enabled or not self.route_manager or not self.vehicle:
            return {
                'distance_to_target': float('inf'),
                'direction_to_target': (0, 0),
                'route_completion': 0.0,
                'target_reached': False,
                'route_complete': False
            }
            
        # Get vehicle location and rotation
        vehicle_loc = self.vehicle.get_location()
        vehicle_rot = self.vehicle.get_transform().rotation
        
        # Get distance and direction to current target
        distance = self.route_manager.get_distance_to_target(vehicle_loc)
        direction = self.route_manager.get_direction_to_target(vehicle_loc, vehicle_rot)
        
        # Calculate route completion percentage
        if len(self.route_manager.waypoints) > 0:
            completion = self.route_manager.current_target_idx / len(self.route_manager.waypoints)
        else:
            completion = 0.0
            
        # Check if we've reached the current target
        target_reached = False
        if distance < config.WAYPOINT_THRESHOLD:
            target_reached = True
            
        # Check if we've completed the entire route
        route_complete = self.route_manager.is_route_complete()
        
        return {
            'distance_to_target': distance,
            'direction_to_target': direction,
            'route_completion': completion,
            'target_reached': target_reached,
            'route_complete': route_complete
        }
    
    def visualize_route(self):
        """Visualize the current route in the simulator"""
        if not self.route_visualization or not self.route_manager or not self.world:
            return
            
        # Get the debug helper
        debug = self.world.debug
        
        # Draw the waypoints
        self.route_manager.draw_waypoints(debug)
    
    def set_navigation_difficulty(self, difficulty):
        """Set the difficulty level for navigation routes"""
        if not self.route_manager:
            return
            
        self.current_difficulty = difficulty 