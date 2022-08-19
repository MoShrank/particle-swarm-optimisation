from math import ceil, floor
from typing import List

import numpy as np


class Particle:
    """
    Particle class that is used in PSO
    """

    def __init__(self):
        """
        Args:
            position (np.array): current position within the search space
            p_best (np.array): own best known position within the search space
            p_best_value (np.array): corresponding fitness value to best position
            velocity (np.array): velocity that defines movement through search space
                                    in each iteration
        """
        self.position = None
        self.p_best = self.position
        self.p_best_value = float("inf")
        self.velocity = None

    def clip_position(self, position: np.ndarray, bounds: List[tuple]) -> np.ndarray:
        """
        Clip position array between given bounds. If position is within bounds
        nothing happens, otherwise the position is clipped to the value of defined
        by bounds.

        Args:
            position (np.array): position within search space
            bounds (array[tuple]): an array which size is equal to the
                                    dimension of the search space.
                                    Each item is a tuple containing
                                    lower and upper bounds.

        Returns:
            new position (np.array): clipped position
        """
        low, high = [e[0] for e in bounds], [e[1] for e in bounds]
        return np.clip(position, low, high)

    def move(self, new_velocity: np.ndarray, bounds: List[tuple]) -> None:
        """
        Moves particle through search space and sets new position
        and velocity. If corresponding type is discrete, the new value
        is rounded to the next integer.

        Args:
            new_velocity (np.array): new velocity
            bounds (array[tuple]): bounds in which new position should be kept
        Returns
            None
        """
        new_position = self.position + new_velocity
        new_position_clipped = self.clip_position(new_position, bounds)
        self.position = new_position_clipped
        self.velocity = new_velocity

    def get_values(self) -> dict:
        """
        Returns a dict of values for evaluating performance

        Args:
            None

        Returns:
            values (dict): current values from object
        """
        return {
            "position": self.position,
            "p_best": self.p_best,
            "p_best_value": self.p_best_value,
            "velocity": self.velocity,
        }
