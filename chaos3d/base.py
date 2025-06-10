"""
base.py

This module defines the AbstractSimulation class, an abstract base class for numerical simulations
of dynamical systems. It provides a framework for integrating systems of ordinary differential
equations (ODEs) using scipy's solve_ivp, and requires subclasses to implement visualization
methods for the simulation results.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp

class AbstractSimulation(ABC):
    """
    Abstract base class for numerical simulations of dynamical systems.
    This class provides a framework for defining and running simulations of
    dynamical systems using the `scipy.integrate.solve_ivp` function.
    Subclasses must implement the `plot_solution` method to visualize the results.
    Attributes:
        f (callable): The system of differential equations to be solved.
        t_span (tuple): The time span for the simulation, defined as (t0, tf).
        t_eval (numpy.ndarray): The time points at which to store the solution.
        X0 (list or numpy.ndarray): Initial conditions for the system.
        method (str): The integration method to use, e.g., 'RK45', 'RK23', etc.
        solution (scipy.integrate.OdeResult): The result of the integration.
    """

    def __init__(self, system, t_span, n_t_steps, X0, method="RK45"):
        """
        Initializes the simulation with the given system, time span, number of time steps,
        initial conditions, and integration method.
        Args:
            system (callable): The system of differential equations to be solved.
            t_span (tuple): The time span for the simulation, defined as (t0, tf).
            n_t_steps (int): The number of time steps to evaluate the solution.
            X0 (list or numpy.ndarray): Initial conditions for the system.
            method (str): The integration method to use, e.g., 'RK45', 'RK23', etc.
        """

        # Ensure the system is a callable function or class instance
        assert callable(system), "system must be a callable function or class instance"
        # Ensure t_span is a tuple/list of two elements (t0, tf) with tf > t0
        assert isinstance(t_span, (tuple, list)) and len(t_span) == 2 and t_span[1] > t_span[0], \
            "t_span must be a tuple or list of two elements (t0, tf) with tf > t0"
        # Ensure n_t_steps is a positive integer
        assert isinstance(n_t_steps, int) and n_t_steps > 0, "n_t_steps must be a positive integer"
        # Ensure X0 is a non-empty iterable
        assert isinstance(X0, (list, tuple, np.ndarray)) and len(X0) > 0, "X0 must be a non-empty iterable"
        # Ensure method is one of the allowed integration methods
        assert method in ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"], \
            "method must be one of 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', or 'LSODA'"

        self.f = system  # System of ODEs
        self.t_span = t_span  # (t0, tf)
        self.t_eval = np.linspace(*t_span, n_t_steps)  # Time points for evaluation
        self.X0 = np.asarray(X0, dtype=float)  # Initial conditions as numpy array
        self.method = method  # Integration method
        self.solution = None  # Will hold the integration result
    
    def __repr__(self):
        cls = self.__class__.__name__
        params = self.f.__dict__  # Get parameters from the system function/class
        return f"<{cls} {params} t_span={self.t_span} n_t_steps={len(self.t_eval)} X0={self.X0} method='{self.method}'>"

    @abstractmethod
    def plot_solution(self, backend='matplotlib'):
        """
        Abstract method to plot the solution of the simulation.
        Args:
            backend (str): The plotting backend to use, e.g., 'matplotlib' or 'plotly'.
        Raises:
            ValueError: If the solution has not been computed yet or if the backend is invalid.
        """
        pass

    @abstractmethod
    def generate_animation(self, fps=30, bitrate=1800):
        """
        Abstract method to generate an animation of the simulation results.
        Args:
            fps (int): Frames per second for the animation.
            bitrate (int): Bitrate for the video encoding.
        Raises:
            ValueError: If the solution has not been computed yet.
        """
        pass

    @property
    def trajectory(self):
        # Return the time points and solution if simulation has been run
        if self.solution is None:
            raise RuntimeError("Simulation not yet run")
        return self.solution.t, self.solution.y

    def run(self):
        """
        Runs the simulation by integrating the system of differential equations
        over the specified time span and initial conditions.
        Returns:
            scipy.integrate.OdeResult: The result of the integration, containing
            the time points and the solution at those points.
        """
        # Call solve_ivp to integrate the system
        self.solution = solve_ivp(
            self.f,
            self.t_span,
            self.X0,
            t_eval=self.t_eval,
            method=self.method
        )
        return self.solution

    def save(self, path):
        """
        Saves the simulation results to a file in NumPy's .npz format.
        Args:
            path (str): The file path where the results will be saved.
        Raises:
            ValueError: If the solution has not been computed yet.
        """
        # Save only if the solution exists
        if self.solution:
            np.savez(path, t=self.solution.t, X=self.solution.y)
    
    @classmethod
    def from_parameters(cls, params, t_span, n_t_steps, X0, method="RK45"):
        # Alternate constructor using a dictionary of parameters
        return cls(**params, t_span=t_span, n_t_steps=n_t_steps, X0=X0, method=method)
