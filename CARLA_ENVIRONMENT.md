# CARLA Reinforcement Learning Environment

This document provides a detailed explanation of the custom Gym environment created for training reinforcement learning agents in the CARLA autonomous driving simulator.

## Sensor Configuration

The environment utilizes three primary sensors attached to the ego vehicle:

1. **RGB Camera**
   - Resolution: 84x84 pixels
   - Field of View: 90 degrees
   - Position: Mounted 2.4m above vehicle, 1.5m forward
   - Purpose: Provides visual input as the main observation for the agent

2. **Collision Sensor**
   - Detects collisions with any objects in the environment
   - Records collision intensity and frame
   - Used for terminal conditions and penalty rewards

3. **Lane Invasion Sensor**
   - Detects when the vehicle crosses lane markings
   - Used for reward calculation and driving metrics

## Observation Space

The observation space consists of:
- RGB images from the front-facing camera (84x84x3)
- Normalized pixel values between 0-255

This raw visual input is preprocessed before being fed to neural networks:
- Normalized to float values between 0-1
- Converted to PyTorch tensors
- Permuted to channel-first format (C, H, W)

## Action Space

The environment uses a discrete action space with 7 possible actions:

| Action Index | Control [Throttle/Brake, Steering] | Description      |
|--------------|----------------------------------|------------------|
| 0            | [0.0, 0.0]                       | No action        |
| 1            | [1.0, 0.0]                       | Throttle         |
| 2            | [0.0, -0.5]                      | Left             |
| 3            | [0.0, 0.5]                       | Right            |
| 4            | [0.5, -0.5]                      | Throttle + Left  |
| 5            | [0.5, 0.5]                       | Throttle + Right |
| 6            | [-0.5, 0.0]                      | Brake            |

Throttle values are positive (0 to 1), brake values are negative (0 to -1), and steering values range from -1 (full left) to 1 (full right).

## Reward Structure

The reward function combines several components to encourage desired driving behavior:

### Positive Rewards
- **Forward Movement**: +0.5 for moving forward (REWARD_FORWARD)
- **Speed Reward**: Up to +0.2 based on current speed, capped at 30 km/h (REWARD_SPEED * min(speed, 30) / 30.0)
- **Target Reached**: +100.0 when reaching target destination (not fully implemented in current version)

### Negative Rewards (Penalties)
- **Collision Penalty**: -100.0 upon collision with any object (REWARD_COLLISION)
- **Lane Invasion Penalty**: -5.0 when crossing lane markings (REWARD_LANE_INVASION)
- **Centerline Distance Penalty**: Proportional to distance from lane center (-0.1 * distance_to_center)
- **Time Penalty**: -0.1 per timestep to encourage efficient completion (REWARD_TIME_PENALTY)
- **Stuck Penalty**: -1.0 if the vehicle hasn't moved significantly in the last 10 steps

The total reward for each step is the sum of all applicable components, creating a shaped reward that balances safe driving with making progress toward goals.

## Terminal Conditions

Episodes end when any of these conditions are met:

1. **Collision**: The vehicle collides with any object
2. **Stuck**: The vehicle moves less than 0.5 meters over 10 timesteps after at least 50 steps
3. **Target Reached**: The vehicle reaches the target destination (when implemented)
4. **Max Steps**: The episode reaches the maximum number of steps (default: 1000)

## Performance Metrics

The environment tracks several metrics to evaluate agent performance:

1. **Total Reward**: Cumulative reward over the episode
2. **Episode Length**: Number of steps per episode
3. **Collision Count**: Total number of collisions
4. **Lane Invasion Count**: Number of times the vehicle crossed lane markings
5. **Average Centerline Deviation**: Mean distance from the lane center
6. **Junction Actions**: Distribution of actions taken at intersections
   - Left turns
   - Right turns
   - Forward movement
   - Other actions

These metrics are used for comparing different algorithms and tracking training progress.

## Environment Interface

The environment implements the standard Gym interface:

- `reset()`: Respawns the vehicle and returns the initial observation
- `step(action)`: Applies the action, advances simulation, and returns (observation, reward, done, truncated, info)
- `render()`: Not explicitly implemented as CARLA handles visualization
- `close()`: Cleans up resources and disconnects from the CARLA server

The `info` dictionary includes additional debugging information:
- Current step
- Total reward
- Vehicle state (position, velocity, speed)
- Distance to centerline
- Whether the vehicle is at a junction

## CARLA Server Integration

The environment connects to a running CARLA server with:
- Synchronous mode enabled (20 FPS)
- Fixed delta seconds (0.05s)
- Configurable weather conditions
- Town01 map by default

The simulation setup ensures that actions and observations are properly synchronized for effective reinforcement learning. 