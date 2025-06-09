def plot_xyz(x, y, z, backend='matplotlib'):
    
    """
    Plot the 3D trajectory of the system in the x, y, z space.
    Parameters:
    x : np.ndarray
        The x-coordinates of the trajectory.
    y : np.ndarray
        The y-coordinates of the trajectory.
    z : np.ndarray
        The z-coordinates of the trajectory.
    backend : str
        The plotting backend to use ('matplotlib' or 'plotly').
    """
    
    if not backend in ['matplotlib', 'plotly']:
        raise ValueError("backend must be 'plotly' or 'matplotlib'")
    
    if backend == 'matplotlib':
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()
        
    elif backend == 'plotly':
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines')])
        fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Z-axis'))
        fig.show()