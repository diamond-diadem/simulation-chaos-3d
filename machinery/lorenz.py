from .base import AbstractSimulation

class LorenzSystem:

    """
    Class representing the Lorenz system of differential equations.
    The Lorenz system is defined by the equations:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    where sigma, rho, and beta are parameters of the system.
    """

    def __init__(self, sigma, rho, beta):

        """Initialize the Lorenz system with parameters sigma, rho, and beta."""

        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def __call__(self, t, X):

        """
        Compute the derivatives of the Lorenz system at time t for state X.
        Parameters:
        t : float
            The current time (not used in the equations but required by the ODE solver).
        X : list or np.ndarray
            The current state of the system, where X[0] = x, X[1] = y, X[2] = z.
        Returns:
        list
            The derivatives [dx/dt, dy/dt, dz/dt] of the Lorenz system.
        """

        x, y, z = X
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return [dx, dy, dz]

class LorenzSimulation(AbstractSimulation):

    """
    Class for simulating the Lorenz system of differential equations.
    Inherits from AbstractSimulation and uses the LorenzSystem class to define the system.
    """

    def __init__(self, sigma, rho, beta, t_span, n_t_steps, X0, method="RK45"):

        """
        Initialize the Lorenz simulation with parameters sigma, rho, beta,
        time span, number of time steps, initial conditions, and integration method.
        Parameters:
        sigma : float
            The sigma parameter of the Lorenz system.
        rho : float
            The rho parameter of the Lorenz system.
        beta : float
            The beta parameter of the Lorenz system.
        t_span : tuple
            The time span for the simulation, defined as (t0, tf).
        n_t_steps : int
            The number of time steps to evaluate the solution.
        X0 : list or np.ndarray
            Initial conditions for the system, should be a 3-element iterable.
        method : str
            The integration method to use, e.g., 'RK45', 'RK23', etc.
        """
        
        assert isinstance(sigma, (int, float)), "sigma must be a number"
        assert isinstance(rho, (int, float)), "rho must be a number"
        assert isinstance(beta, (int, float)), "beta must be a number"

        system  = LorenzSystem(sigma, rho, beta)
        super().__init__(system, t_span, n_t_steps, X0, method)

    def plot_solution(self, backend='matplotlib'):

        """
        Plot the solution of the Lorenz system using the specified backend.
        Parameters:
        backend : str
            The plotting backend to use, either 'matplotlib' or 'plotly'.
        Raises:
        ValueError: If the solution has not been computed yet or if an invalid backend is specified.
        """

        assert backend in ['matplotlib', 'plotly'], "backend must be 'plotly' or 'matplotlib'"
        
        if self.solution is None:
            raise ValueError("No solution available. Please run the simulation first.")
        
        x, y, z = self.solution.y
        
        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, lw=1, color='blue')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()
        
        elif backend == 'plotly':
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(width=1, color='blue')
            ))
            fig.update_layout(scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ))
            fig.show()