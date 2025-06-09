from .base import AbstractSimulation

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

class LorenzSimulation(AbstractSimulation):
    
    def __init__(self, sigma, rho, beta, t_span, n_t_steps, X0, method="RK45"):

        system  = LorenzSystem(sigma, rho, beta)
        super().__init__(system, t_span, n_t_steps, X0, method)

    def plot_solution(self, backend='matplotlib'):
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