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
    
    # Check if the backend argument is valid
    if not backend in ['matplotlib', 'plotly']:
        raise ValueError("backend must be 'plotly' or 'matplotlib'")
    
    if backend == 'matplotlib':
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
        ax.plot(x, y, z)  # Plot the trajectory
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()  # Display the plot
        
    elif backend == 'plotly':
        import plotly.graph_objects as go
        
        # Create a 3D scatter plot with lines connecting the points
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines')])
        fig.update_layout(
            scene=dict(
                xaxis_title='X-axis',
                yaxis_title='Y-axis',
                zaxis_title='Z-axis'
            )
        )
        fig.show()  # Display the plot