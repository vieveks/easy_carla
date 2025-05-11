#!/usr/bin/env python
"""
Simple script to visualize the camera input from the CARLA environment.
This is useful for debugging and understanding what the agent "sees".
"""
import os
import sys
import time
import argparse
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
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
    parser = argparse.ArgumentParser(description="CARLA Camera Visualization")
    
    # Environment parameters
    parser.add_argument('--carla_host', type=str, default=config.CARLA_HOST,
                        help='CARLA server host')
    parser.add_argument('--carla_port', type=int, default=config.CARLA_PORT,
                        help='CARLA server port')
    parser.add_argument('--carla_map', type=str, default=config.CARLA_MAP,
                        help='CARLA map to use')
    
    # Visualization parameters
    parser.add_argument('--camera_width', type=int, default=640,
                        help='Width of the camera window')
    parser.add_argument('--camera_height', type=int, default=480,
                        help='Height of the camera window')
    parser.add_argument('--fps', type=int, default=20,
                        help='Target FPS for the visualization')
    
    # Simulation parameters
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration to run the visualization in seconds')
    parser.add_argument('--manual_control', action='store_true',
                        help='Enable manual control (otherwise vehicle moves randomly)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create environment with larger image shape for better visualization
    env = CarlaEnv(
        host=args.carla_host,
        port=args.carla_port,
        town=args.carla_map,
        image_shape=(args.camera_height, args.camera_width, 3)
    )
    
    # Create camera viewer
    camera_viewer = CameraViewer(
        window_name="CARLA Camera View",
        width=args.camera_width,
        height=args.camera_height
    )
    camera_viewer.fps = args.fps
    camera_viewer.start()
    
    try:
        logger.info("Starting camera visualization")
        logger.info("Press Ctrl+C to exit")
        
        # Reset environment to initialize
        state = env.reset()
        camera_viewer.update(state)
        
        # Start time
        start_time = time.time()
        frame_count = 0
        
        if args.manual_control:
            logger.info("Manual control mode: Use WASD to drive")
            # TODO: Implement manual control
            # For now, just do random actions
            manual_mode = False
        else:
            manual_mode = False
        
        # Main loop
        while time.time() - start_time < args.duration:
            # Select random action if not manual
            if not manual_mode:
                action = np.random.randint(0, len(config.DISCRETE_ACTIONS))
            else:
                # TODO: Get action from keyboard input
                action = 1  # Default: move forward
            
            # Apply action to environment
            next_state, reward, done, info = env.step(action)
            
            # Update camera viewer
            camera_viewer.update(next_state)
            
            # Count frames for FPS calculation
            frame_count += 1
            
            # If episode ended, reset
            if done:
                state = env.reset()
                camera_viewer.update(state)
            else:
                state = next_state
            
            # Small sleep to not overwhelm the system
            time.sleep(1.0 / args.fps)
        
        # Calculate actual FPS
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        logger.info(f"Visualization complete: {frame_count} frames in {elapsed_time:.2f}s ({actual_fps:.2f} FPS)")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        camera_viewer.stop()
        env.close()
        logger.info("Environment closed")

if __name__ == "__main__":
    main() 