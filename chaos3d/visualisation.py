"""
visualisation.py

This module provides functions for visualizing and animating 3D trajectories.
It supports both matplotlib and plotly backends for static 3D plotting, and
offers animation generation and saving using matplotlib.

Functions:
- plot_xyz: Plot a 3D trajectory using matplotlib or plotly.
- generate_animation_txyz: Generate and save a 3D trajectory animation as a video file.
"""

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

def generate_animation_txyz(t, x, y, z, filename, fps=30, bitrate=1800):
    """
    Generate an animation of the trajectory in 3D space and save it as a video file.
    Parameters:
    t : np.ndarray
        The time points of the trajectory.
    x : np.ndarray
        The x-coordinates of the trajectory.
    y : np.ndarray
        The y-coordinates of the trajectory.
    z : np.ndarray
        The z-coordinates of the trajectory.
    filename : str
        The name of the file to save the animation.
    fps : int
        Frames per second for the animation.
    bitrate : int
        Bitrate for the video encoding.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FFMpegWriter

    # Initialize the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Configure the view
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_zlim([min(z), max(z)])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

     # Set camera to view from (10, 10, 10) toward the origin
    ax.view_init(elev=35.264, azim=45)

    # Line for the trajectory and point for the current position
    line, = ax.plot([], [], [], lw=2)
    point, = ax.plot([], [], [], 'ro')  # red point

    # Initialization function
    def init():
        """Initialize the line and the point."""
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    # Animation function
    def update(frame):
        """Update the line and the point for each frame."""
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])

        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])

        return line, point

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init,
                        interval=18, blit=False)

    # Define the video writer
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)

    # Save as MP4
    ani.save(filename, writer=writer)

    # Close the figure to prevent the last frame from being displayed in a Jupyter notebook
    plt.close(fig)