from .base import AbstractSimulation

class RosslerSystem:

    """
    Class representing the Rossler system of differential equations.
    The Rossler system is defined by the equations:
    dx/dt = -y - z
    dy/dt = x + a * y
    dz/dt = b + z * (x - c)
    where a, b, and c are parameters of the system.
    """

    def __init__(self, a, b, c):

        """Initialize the Rossler system with parameters a, b, and c."""
        
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, t, X):

        """
        Compute the derivatives of the Rossler system at time t for state X.
        Parameters:
        t : float
            The current time (not used in the equations but required by the ODE solver).
        X : list or np.ndarray
            The current state of the system, where X[0] = x, X[1] = y, X[2] = z.
        Returns:
        list
            The derivatives [dx/dt, dy/dt, dz/dt] of the Rossler system.
        """

        x, y, z = X
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return [dx, dy, dz]
    
class RosslerSimulation(AbstractSimulation):

    """
    Class for simulating the Rossler system of differential equations.
    Inherits from AbstractSimulation and uses the RosslerSystem class to define the system.
    """
    def __init__(self, a, b, c, t_span, n_t_steps, X0, method="RK45"):

        """
        Initialize the Rossler simulation with parameters a, b, c,
        time span, number of time steps, initial conditions, and integration method.
        Parameters:
        a : float
            The a parameter of the Rossler system.
        b : float
            The b parameter of the Rossler system.
        c : float
            The c parameter of the Rossler system.
        t_span : tuple or list
            The time span for the simulation, defined as (t0, tf).
        n_t_steps : int
            The number of time steps to evaluate the solution.
        X0 : list or np.ndarray
            Initial conditions for the system, where X0[0] = x, X0[1] = y, X0[2] = z.
        method : str
            The integration method to use, e.g., 'RK45', 'RK23', etc.
        """

        assert isinstance(a, (int, float)), "a must be a number"
        assert isinstance(b, (int, float)), "b must be a number"
        assert isinstance(c, (int, float)), "c must be a number"

        system = RosslerSystem(a, b, c)
        super().__init__(system, t_span, n_t_steps, X0, method)

    def plot_solution(self, backend='matplotlib'):

        """
        Plots the solution of the Rossler system in 3D.
        Args:
            backend (str): The plotting backend to use, either 'matplotlib' or 'plotly'.
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