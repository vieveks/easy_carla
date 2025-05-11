"""
Utility functions and tools for the CARLA reinforcement learning project.
"""

import os

def ensure_directory(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to create
    """
    os.makedirs(directory, exist_ok=True)
    return directory
