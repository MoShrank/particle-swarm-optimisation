import logging
import random
from typing import Callable, List, Tuple, TypedDict, Union

import numpy as np
from tqdm import tqdm
from util.parallelization import try_parallelization

from .particle import Particle

RealNumeric = Union[float, int]

ParamType = TypedDict("ParamType", {"name": str, "bounds": Tuple[RealNumeric], "type": str})

logger = logging.getLogger("main")


class PSO:
    """
    Particle swarm optimisation for minimizing/maximizing a fitness in an n-dimensional search space.
    It works for both discrete and continuous values given that the type of parameter
    is passed within the search_space dictionary. Algorithm implements early stopping to stop
    if g_best_value has not been improved by at least 0.0003 on average in a given amount of
    iterations.

    Attributes:
        fitness (Callable): function to be optimised
        number_of_particles (int): number of particle that move through search space
        number_of_iterations (int): number iterations after which its terminated
        g_best (numpy.array): best know position of swarm in search space
        g_best_value (float): associated fitness value to g_best
        particles (List[Particle]): list that contains all particles
        parameter_names (List[str]): list of parameter names
        bounds (List[Tuple]): list that contains lower and upper bound for each parameter
        types (List[str]): list that contains parameter types
        number_of_dimensions (int): number of dimensions which is the same as the number of
                                    parameters
        w (RealNumber): inertia weight to scale exploration and exploitation
        c_1 (RealNumber): weight for how much p_best is considered when moving through space
        c_2 (RealNumber): weight for much g_best is considered when moving through space
        max_jobs (int): maximum number of jobs to run in parallel
        stop_iter (int): number of iterations after which its terminated if g_best has not been
                        improved by at least 0.0003
        g_best_past_values (List[float]): list of past g_best values to keep track of average improvement
        maximize (bool): parameter to maximize instead of minimizing fitness function
        iteration_results (pandas.df): dataframe that keeps track of particle values in each iteration
    """

    def __init__(
        self,
        fitness: Callable,
        search_space: List[ParamType],
        number_of_particles: int = 4,
        number_of_iterations: int = 10,
        stop_iter: int = 10,
        w: RealNumeric = 4,
        c_1: RealNumeric = 2,
        c_2: RealNumeric = 2,
        max_jobs: int = 1,
        maximize: bool = False,
    ):
        """
        Args:
            fitness (Callable): fitness function that takes n-dimensional array of parameters
                                and returns fitness score
            search_space (List[ParamType]): List of parameters which is used to set bounds and types list
            number_of_particles (int): number of particle that move through search space
            number_of_iterations (int): number iterations after which its terminated
            stop_iter (int): number of iterations after which its terminated if g_best has not been
                            improved by at least 0.0003
            w (RealNumber): initial scaling for exploration and exploitation
            c_1 (RealNumber): initial weight for how much p_best is considered when moving through space
            c_2 (RealNumber): initial weight for much g_best is considered when moving through space
            max_jobs (int): maximum number of jobs to run in parallel
            maximize (bool): maximize fitness function
        """
        self.fitness = fitness
        self.number_of_particles = number_of_particles
        self.number_of_iterations = number_of_iterations

        self.g_best = None
        self.g_best_value = float("inf")

        self.particles = [Particle() for _ in range(number_of_particles)]

        self.search_space = search_space

        self.bounds = [parameter["bounds"] for parameter in search_space]
        self.types = [parameter["type"] for parameter in search_space]

        self.number_of_dimensions = len(search_space)

        self.w = w
        self.c_1 = c_1
        self.c_2 = c_2
        self.max_jobs = max_jobs

        self.stop_iter = stop_iter
        self.g_best_past_values = []

        self.maximize = maximize

        self.iteration_results = []

    def update_velocity(
        self,
        current_velocity: np.ndarray,
        current_position: np.ndarray,
        p_best: np.ndarray,
    ) -> np.ndarray:
        """
        Function that updates velocity. Based on this formula:
            V_new = w * V_old +
                    c_1 * r(0, 1) * (p_best - current_position) +
                    c_2 * r(0, 1) * (g_best - current_position)

            w = inertia weight
            c_1 = personal weight
            c_2 = social weight
            r(0, 1) = random uniform number between 0 and 1

        Args:
            current_velocity (np.ndarray): current velocity of particle
            currenct_position (np.ndarray): current position of particle
            p_best (np.ndarray): current own best known position of particle
        Returns:
            new_velocity (np.ndarray): updated velocity
        """
        r_1 = random.uniform(0, 1)
        r_2 = random.uniform(0, 1)
        p_term = self.c_1 * r_1 * (p_best - current_position)
        g_term = self.c_2 * r_2 * (self.g_best - current_position)
        new_velocity = (self.w * current_velocity) + p_term + g_term
        return new_velocity

    # TODO: should probably accept mode instead of fitness function
    def fitness_wrapper(self, values: List[RealNumeric]) -> float:
        """
        Wrapper around actual fitness function to pass parameter values by name and return negative
        value if we want to maximise our function.

        Args:
            values (List[RealNumeric]): list of values to be passed to fitness function
        Returns:
            fitness_value (float): result of fitness function
        """
        kwargs = {}

        # TODO: find out if this is the best way to go for
        # categorical/discrete values, one way to change it is
        # to improve the threshol of when it will be rounded up
        # or down

        for value, param in zip(values, self.search_space):

            # the value to be passed to our fitness function
            # will be overriden if parameter is continuous or categorical
            argument_value = value

            # check if type is not continuous to round it if needed
            if param["type"] in ["categorical", "discrete"]:
                rounded_value = round(argument_value)

                # get categorical value or just assign rounded discrete value
                if param["type"] == "categorical":
                    argument_value = param["values"][value]
                else:
                    argument_value = rounded_value

            kwargs[param["name"]] = argument_value

        score = self.fitness(kwargs)

        return -score if self.maximize else score

    def update_coef(self, iteration: int) -> None:
        """
        Updates weights in class.

        Args:
            iteration (int): number of iteration to consider in weight decay
        """
        t = iteration
        n = self.number_of_iterations

        self.w = (0.4 / n ** 2) * (t - n) ** 2 + 0.4
        self.c_1 = -3 * t / n + 3.5
        self.c_2 = 3 * t / n + 0.5

    def init_particles(self) -> None:
        """
        Initialises particle velocity and position. Position is uniformly drawn
        from within the bounds of each parameter. Velocity is initialised as zero.
        """
        for particle in self.particles:

            position = []

            for (lower, upper), parameter_type in zip(self.bounds, self.types):
                if parameter_type == "continuous":
                    position.append(np.random.uniform(lower, upper))
                else:
                    # add +1 because upper bound is exclusive
                    position.append(np.random.randint(lower, upper + 1))

            # init position values as object to mix int and float values
            particle.position = np.array(position, dtype=object)
            particle.velocity = np.zeros(self.number_of_dimensions)

    def get_average_g_best_improvement(self) -> float:
        """
        Calculates average improvement between each neighbour in a list of g_best values.

        Returns:
            avg_dif (float): average improvement
        """

        # all difference values will have length of len(list) - 1
        differences = []
        for ind, val in enumerate(self.g_best_past_values):

            # if current index is last index there is no more neightbour to the right
            last_index = len(self.g_best_past_values) - 1
            if ind < last_index:

                # calculate difference to neighbour to the right
                difference = abs(val - self.g_best_past_values[ind + 1])
                differences.append(difference)

        avg_dif = sum(differences) / len(differences)

        return avg_dif

    def get_position_tuple(self, position: List[RealNumeric]) -> List[Tuple]:
        """
        Creates a list of position tuples which contains the parameters name and its value.
        The value can be any type. Either a string or a number.

        Args:
            position (List[RealNumeric]): a list of position values for example g_best

        Returns:
            position tuple (List[Tuple]): list of tuples in the form of (parameter_name, parameter value)
        """
        positions = []

        for param, position in zip(self.search_space, position):
            if param["type"] == "categorical":
                positions.append((param["name"], param["values"][position]))
            else:
                positions.append((param["name"], position))

        return positions

    def save_particle_data(
        self, particle: Particle, iteration: int, particle_index: int, current_fitness_value: int
    ) -> None:
        """
        Saves data about current iteration and particle in a list of dictionaries.

        Args:
            particle (Particle): particle instance
            iteration (int): current iteration
            particle_index (int): corresponding particle index to particle instance
        """
        particle_results = particle.get_values()
        particle_results["iteration"] = iteration
        particle_results["g_best_value"] = self.g_best_value
        particle_results["particle_index"] = particle_index
        particle_results["current_fitness_value"] = current_fitness_value
        particle_results["position"] = self.get_position_tuple(particle_results["position"])
        particle_results["g_best"] = self.get_position_tuple(self.g_best)
        particle_results["p_best"] = self.get_position_tuple(particle_results["p_best"])

        self.iteration_results.append(particle_results)

    def optimize(self) -> List[dict]:
        """
        Runs optimisiation based on class attributes. It will try to parallelize
        each iteration if max_jobs is higher than 1 and its possible. Early stopping
        is used to terminate the optimisation if the average improvement
        over a given amount of iterations is not higher than a certain threshold.

        Returns:
            iterations_results List(dict): each dict contains information
                                            indexed by iteration + particle_index
        """
        # init position and velocity of particles
        self.init_particles()

        for iteration in tqdm(range(self.number_of_iterations), total=self.number_of_iterations):

            logger.info(f"Iteration: {iteration}")

            # get position values from particles for parallelization
            params_list = [particle.position for particle in self.particles]
            # execute fitness function in parallel
            results = try_parallelization(self.fitness_wrapper, params_list, max_jobs=self.max_jobs)

            # update g_best and p_best values
            for particle, fitness_value in zip(self.particles, results):
                p_best_value = particle.p_best_value
                current_position = particle.position

                if p_best_value > fitness_value:
                    particle.p_best_value = fitness_value
                    particle.p_best = current_position

                if self.g_best_value > fitness_value:
                    self.g_best_value = fitness_value
                    self.g_best = current_position

            self.update_coef(iteration)

            # save values and update position of particles
            for particle_index, particle in enumerate(self.particles):
                # save particle info about iteration and particle
                self.save_particle_data(
                    particle, iteration, particle_index, results[particle_index]
                )

                # obtain and set new velocity
                new_velocity = self.update_velocity(
                    particle.velocity, particle.position, particle.p_best
                )
                particle.move(new_velocity, self.bounds)

            logger.info(f"Current g_best_value: {self.g_best_value}")

            # check if enough iterations have been made
            if len(self.g_best_past_values) == self.stop_iter:
                # get average differemce
                avg_dif = self.get_average_g_best_improvement()

                # return if avg_dif is less than threshold
                if avg_dif < 0.0003:
                    return self.iteration_results

                # pop oldest value because we only want to keep track of
                # last 'stop_iter' values
                self.g_best_past_values.pop(0)

            self.g_best_past_values.append(self.g_best_value)

        return self.iteration_results
