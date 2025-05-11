"""
Configuration file for the Carla RL project
"""

# Carla connection parameters
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0
CARLA_MAP = "Town01"
WEATHER = 'ClearNoon'  # Options: 'ClearNoon', 'CloudyNoon', 'WetNoon', 'HardRainNoon'

# Environment parameters
FPS = 20  # Frames per second
FIXED_DELTA_SECONDS = 1.0 / FPS
MAX_EPISODE_STEPS = 1000
SENSORS = {
    'rgb_camera': {
        'image_size_x': 84,
        'image_size_y': 84,
        'fov': 90,
        'position_x': 1.5,
        'position_y': 0.0,
        'position_z': 2.4,
        'rotation_pitch': 0,
        'rotation_roll': 0,
        'rotation_yaw': 0
    },
    'collision': {},
    'lane_invasion': {}
}

# Vehicle parameters
VEHICLE_TYPE = 'vehicle.tesla.model3'
SPAWN_POINT_INDEX = 1  # The index of the spawn point to use

# Action space parameters
DISCRETE_ACTIONS = {
    0: [0.0, 0.0],    # No action
    1: [1.0, 0.0],    # Throttle
    2: [0.0, -0.5],   # Left
    3: [0.0, 0.5],    # Right
    4: [0.5, -0.5],   # Throttle + Left
    5: [0.5, 0.5],    # Throttle + Right
    6: [-0.5, 0.0],   # Brake
}

# Reward parameters
REWARD_FORWARD = 0.5
REWARD_COLLISION = -100.0
REWARD_LANE_INVASION = -5.0
REWARD_CENTERLINE_DISTANCE = -0.1  # Coefficient for distance to centerline
REWARD_SPEED = 0.2  # Coefficient for speed reward
REWARD_TARGET_REACHED = 100.0
REWARD_TIME_PENALTY = -0.1  # Small penalty for each timestep

# Navigation parameters
NAVIGATION_ENABLED = False  # Whether to use waypoint navigation
WAYPOINT_THRESHOLD = 5.0  # Distance in meters to consider a waypoint reached
WAYPOINT_DIRECTION_WEIGHT = 0.5  # Weight for directional reward
WAYPOINT_DISTANCE_WEIGHT = 0.3  # Weight for distance-based reward

# Route parameters
ROUTE_DIFFICULTY = 'easy'  # Options: 'easy', 'medium', 'hard'
DIFFICULTY_PROGRESSION = {  # Episode thresholds for increasing difficulty
    'easy': 0,      # Start with easy
    'medium': 100,  # Switch to medium after 100 episodes
    'hard': 300     # Switch to hard after 300 episodes
}
ROUTE_VISUALIZATION = True  # Whether to visualize the route in the simulator

# Training hyperparameters
COMMON_HYPERPARAMS = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'batch_size': 64,
    'training_episodes': 1000,
    'save_interval': 50,
    'eval_interval': 20,
    'model_dir': 'models/',
    'log_dir': 'logs/',
    'results_dir': 'results/',
}

DQN_HYPERPARAMS = {
    **COMMON_HYPERPARAMS,
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay': 50000,
    'target_update': 1000,
    'memory_size': 100000,
    'hidden_dim': 512,
}

DDQN_HYPERPARAMS = {
    **DQN_HYPERPARAMS,
}

DUELING_DQN_HYPERPARAMS = {
    **DQN_HYPERPARAMS,
}

SARSA_HYPERPARAMS = {
    **COMMON_HYPERPARAMS,
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay': 50000,
    'memory_size': 100000,
    'hidden_dim': 512,
}

PPO_HYPERPARAMS = {
    **COMMON_HYPERPARAMS,
    'clip_param': 0.2,
    'ppo_epochs': 10,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'hidden_dim': 512,
}

# Evaluation parameters
EVAL_EPISODES = 10
RECORD_VIDEO = True 