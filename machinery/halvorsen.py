from .base import AbstractSimulation
from .visualisation import plot_xyz

class HalvorsenSystem:
    """
    Class representing the Halvorsen system of differential equations.
    The Halvorsen system is defined by the equations:
        dx/dt = -a*x - 4*y - 4*z - y^2
        dy/dt = -a*y - 4*z - 4*x - z^2
        dz/dt = -a*z - 4*x - 4*y - x^2
    where 'a' is a parameter of the system.
    """

    def __init__(self, a):
        """Initialize the Halvorsen system with parameter 'a'."""
        self.a = a

    def __call__(self, t, X):
        """
        Compute the derivatives of the Halvorsen system at time t for state X.
        Parameters:
            t : float
                The current time (not used in the equations but required by the ODE solver).
            X : list or np.ndarray
                The current state of the system, where X[0] = x, X[1] = y, X[2] = z.
        Returns:
            list
                The derivatives [dx/dt, dy/dt, dz/dt] of the Halvorsen system.
        """
        x, y, z = X  # Unpack the state vector
        dx = -self.a * x - 4 * y - 4 * z - y**2         # Halvorsen equation for dx/dt
        dy = -self.a * y - 4 * z - 4 * x - z**2         # Halvorsen equation for dy/dt
        dz = -self.a * z - 4 * x - 4 * y - x**2         # Halvorsen equation for dz/dt
        return [dx, dy, dz]

class HalvorsenSimulation(AbstractSimulation):
    """
    Class for simulating the Halvorsen system of differential equations.
    Inherits from AbstractSimulation and uses the HalvorsenSystem class to define the system.
    """

    def __init__(self, a, t_span, n_t_steps, X0, method="RK45"):
        """
        Initialize the Halvorsen simulation with parameter 'a',
        time span, number of time steps, initial conditions, and integration method.
        Parameters:
            a : float
                The parameter 'a' of the Halvorsen system.
            t_span : tuple
                The time span for the simulation, defined as (t0, tf).
            n_t_steps : int
                The number of time steps to evaluate the solution.
            X0 : list or np.ndarray
                Initial conditions for the system, should be a 3-element iterable.
            method : str
                The integration method to use, e.g., 'RK45', 'RK23', etc.
        """
        assert isinstance(a, (int, float)), "a must be a number"

        system  = HalvorsenSystem(a)  # Instantiate the Halvorsen system
        super().__init__(system, t_span, n_t_steps, X0, method)  # Initialize the base simulation

    def plot_solution(self, backend='matplotlib'):
        """
        Plot the solution of the Halvorsen system using the specified backend.
        Parameters:
            backend : str
                The plotting backend to use, either 'matplotlib' or 'plotly'.
        Raises:
            ValueError: If the solution has not been computed yet or if an invalid backend is specified.
        """
        # Use the plot_xyz utility to plot the solution in 3D
        plot_xyz(*self.solution.y, backend=backend)
