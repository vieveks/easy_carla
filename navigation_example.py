#!/usr/bin/env python
"""
Example script for running the CARLA environment with point-to-point navigation.
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("navigation_example.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import configurations and utilities
import config
from carla_env.rl_env import CarlaEnv
from utils.helpers import set_random_seed
from utils.visualization import CameraViewer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CARLA Navigation Example")
    
    # Environment parameters
    parser.add_argument('--carla_host', type=str, default=config.CARLA_HOST,
                        help='CARLA server host')
    parser.add_argument('--carla_port', type=int, default=config.CARLA_PORT,
                        help='CARLA server port')
    parser.add_argument('--carla_map', type=str, default=config.CARLA_MAP,
                        help='CARLA map to use')
    
    # Navigation parameters
    parser.add_argument('--route_difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty level for navigation routes')
    parser.add_argument('--visualize_route', action='store_true', default=True,
                        help='Visualize the navigation route in CARLA')
    parser.add_argument('--waypoint_threshold', type=float, default=5.0,
                        help='Distance threshold to consider a waypoint reached (in meters)')
    
    # Simulation parameters
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Visualization parameters
    parser.add_argument('--show_camera', action='store_true', default=True,
                        help='Show camera view in a separate window')
    parser.add_argument('--camera_width', type=int, default=400,
                        help='Width of the camera window')
    parser.add_argument('--camera_height', type=int, default=300,
                        help='Height of the camera window')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Update config with runtime args
    config.NAVIGATION_ENABLED = True
    config.ROUTE_DIFFICULTY = args.route_difficulty
    config.ROUTE_VISUALIZATION = args.visualize_route
    config.WAYPOINT_THRESHOLD = args.waypoint_threshold
    
    # Create environment
    env = CarlaEnv(
        host=args.carla_host,
        port=args.carla_port,
        town=args.carla_map,
        navigation_enabled=True,
        route_difficulty=args.route_difficulty
    )
    
    # Create camera viewer if enabled
    camera_viewer = None
    if args.show_camera:
        # Detect if running on a server (no display) - use headless mode
        headless = False
        if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
            logger.info("No display detected, using headless mode for camera viewer")
            headless = True
            
        camera_viewer = CameraViewer(
            window_name="CARLA Navigation Camera View",
            width=args.camera_width,
            height=args.camera_height,
            headless=headless
        )
        camera_viewer.start()
    
    try:
        logger.info("Starting navigation example")
        logger.info(f"Using route difficulty: {args.route_difficulty}")
        
        # Run episodes
        for episode in range(args.episodes):
            logger.info(f"Starting episode {episode+1}/{args.episodes}")
            
            # Reset environment
            state = env.reset()
            
            # Update camera viewer with initial state
            if camera_viewer:
                camera_viewer.update(state)
            
            done = False
            total_reward = 0
            step = 0
            
            # Simple policy: always go forward and steer towards the target
            while not done:
                # Get navigation info
                info = env.client.get_navigation_info()
                
                # Decide action based on navigation info
                # Default action: throttle forward
                action = 1  # Forward
                
                # If we need to turn
                if info['direction_to_target'][1] > 0.3:  # Need to turn right
                    action = 5  # Throttle + Right
                elif info['direction_to_target'][1] < -0.3:  # Need to turn left
                    action = 4  # Throttle + Left
                
                # Take action
                next_state, reward, done, step_info = env.step(action)
                
                # Update camera viewer
                if camera_viewer:
                    camera_viewer.update(next_state)
                
                total_reward += reward
                step += 1
                
                # Log progress
                if step % 10 == 0:
                    # Get distance to target
                    distance = info['distance_to_target']
                    
                    # Get target index
                    if env.client.route_manager:
                        target_idx = env.client.route_manager.current_target_idx
                        num_waypoints = len(env.client.route_manager.waypoints)
                        logger.info(f"Step {step}, Target {target_idx+1}/{num_waypoints}, "
                                   f"Distance: {distance:.2f}m, Reward: {reward:.2f}")
            
            # Get episode statistics
            stats = env.get_episode_statistics()
            
            logger.info(f"Episode {episode+1} finished")
            logger.info(f"  Steps: {step}")
            logger.info(f"  Total reward: {total_reward:.2f}")
            logger.info(f"  Targets reached: {stats['targets_reached']}")
            logger.info(f"  Collisions: {stats['collision_count']}")
            logger.info(f"  Lane invasions: {stats['lane_invasion_count']}")
            
            # Wait a bit before starting the next episode
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if camera_viewer:
            camera_viewer.stop()
        env.close()
        logger.info("Environment closed")

if __name__ == "__main__":
    main() 