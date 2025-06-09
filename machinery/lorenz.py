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
    def __init__(self, sigma, rho, beta, t_span, t_eval, X0):
        self.f = LorenzSystem(sigma, rho, beta)
        self.t_span = t_span
        self.t_eval = t_eval
        self.X0 = X0
        self.solution = None

    def run(self):
        self.solution = solve_ivp(
            self.f,
            self.t_span,
            self.X0,
            t_eval=self.t_eval,
            method="RK45"
        )
        return self.solution

    def save(self, path):
        if self.solution:
            np.savez(path, t=self.solution.t, X=self.solution.y)

    def plot_solution(self):

        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')

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
        # ax.plot(x, y, z, lw=0.5)

        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # ax.set_title("Attracteur de Lorenz")

        # plt.show()
