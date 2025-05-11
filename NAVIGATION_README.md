# Point-to-Point Navigation for CARLA RL

This document explains how to use the point-to-point navigation feature for training reinforcement learning agents in the CARLA environment.

## Overview

The point-to-point navigation system enables agents to learn how to drive from a starting point to one or more target destinations. The system generates routes with varying levels of difficulty, visualizes waypoints in the simulator, and provides specialized rewards for successful navigation.

## Quick Start

### 1. Running the Navigation Example

To quickly see the navigation system in action, run the example script:

```bash
python navigation_example.py
```

This script demonstrates point-to-point navigation using a simple rule-based policy that steers toward waypoints.

### 2. Training an RL Agent with Navigation

To train a reinforcement learning agent with navigation:

```bash
python main.py --algorithm dqn --train --episodes 1000 --navigation --route_difficulty easy
```

### 3. Evaluating a Trained Agent

To evaluate a trained agent on navigation tasks:

```bash
python main.py --algorithm dqn --eval --navigation --load_checkpoint models/your_checkpoint.pth
```

## Navigation Parameters

The following command-line arguments control the navigation system:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--navigation` | Enable point-to-point navigation | False |
| `--route_difficulty` | Difficulty level: 'easy', 'medium', or 'hard' | 'easy' |
| `--visualize_route` | Visualize the route in CARLA | True |
| `--waypoint_threshold` | Distance (m) to consider a waypoint reached | 5.0 |

## Camera Visualization

You can now visualize what the agent "sees" in a separate window. This feature shows the camera input that is being used by the agent for decision-making.

To enable camera visualization, use the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--show_camera` | Show camera input in a separate window | False |
| `--camera_width` | Width of the camera window (pixels) | 400 |
| `--camera_height` | Height of the camera window (pixels) | 300 |

Example:

```bash
# Run with camera visualization
python navigation_example.py --show_camera --camera_width 640 --camera_height 480

# Train with camera visualization
python main.py --algorithm dqn --train --navigation --show_camera
```

The camera window provides a real-time view of what the agent is processing, making it easier to understand its behavior and debug issues.

## Difficulty Levels

The navigation system provides three levels of difficulty:

1. **Easy**: 
   - 2-3 waypoints per route
   - Short distances between waypoints (20-40m)
   - Primarily straight roads with minimal turns

2. **Medium**:
   - 4-6 waypoints per route
   - Medium distances between waypoints (50-100m)
   - Includes some turns and junctions

3. **Hard**:
   - 7-10 waypoints per route
   - Longer distances between waypoints (100-200m)
   - Complex routes with multiple turns and junctions

By default, the system starts with 'easy' routes and progressively increases difficulty based on the number of episodes completed:
- Episodes 0-99: Easy routes
- Episodes 100-299: Medium routes
- Episodes 300+: Hard routes

You can override this progression using the `--route_difficulty` parameter.

## Reward Structure

The navigation system enhances the reward function with several navigation-specific components:

1. **Target Reached**: Large positive reward when reaching a waypoint
2. **Direction Reward**: Encourages facing towards the current target
3. **Distance Improvement**: Rewards for getting closer to the target
4. **Route Completion**: Extra large reward for completing the entire route

These rewards work alongside the existing rewards for forward movement, speed, centerline following, and penalties for collisions and lane invasions.

## Implementation Details

The navigation system comprises several components:

1. **RouteManager**: Generates and tracks waypoints and routes
2. **Waypoint Visualization**: Displays waypoints in the simulator with color-coding
3. **Reward Enhancement**: Extends the reward function for navigation tasks
4. **Difficulty Progression**: Automatically adjusts difficulty based on training progress
5. **Camera Visualization**: Shows the agent's perspective in a separate window

## Custom Routes

To create custom routes instead of random ones, you can modify the `generate_random_route` method in `carla_env/navigation.py` to use predefined waypoints. 