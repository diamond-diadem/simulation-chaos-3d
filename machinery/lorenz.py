import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

class LorenzSystem:
    def __init__(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def __call__(self, t, X):
        x, y, z = X
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return [dx, dy, dz]
    
class LorenzSimulation:
    def __init__(self, sigma, rho, beta, t_span, n_t_steps, X0, method="RK45"):
        
        assert isinstance(sigma, (int, float)), "sigma must be a number"
        assert isinstance(rho, (int, float)), "rho must be a number"
        assert isinstance(beta, (int, float)), "beta must be a number"
        assert isinstance(t_span, (tuple, list)) and len(t_span) == 2 and t_span[1] > t_span[0], \
            "t_span must be a tuple or list of two elements (t0, tf) with tf > t0"
        assert isinstance(n_t_steps, int) and n_t_steps > 0, "n_t_steps must be a positive integer"
        assert isinstance(X0, (list, tuple, np.ndarray)) and len(X0) == 3, "X0 must be a 3-element iterable"
        assert method in ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"], \
            "method must be one of 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', or 'LSODA'"

        self.f = LorenzSystem(sigma, rho, beta)
        self.t_span = t_span
        self.t_eval = np.linspace(*t_span, n_t_steps)
        self.X0 = X0
        self.method = method
        self.solution = None

    def run(self):
        self.solution = solve_ivp(
            self.f,
            self.t_span,
            self.X0,
            t_eval=self.t_eval,
            method=self.method,
        )
        return self.solution

    def save(self, path):
        if self.solution:
            np.savez(path, t=self.solution.t, X=self.solution.y)

    def plot_solution(self):

        if self.solution is None:
            raise ValueError("No solution available. Please run the simulation first.")

        x, y, z = self.solution.y
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