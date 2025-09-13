# hologram_utils.py
import numpy as np
import plotly.graph_objects as go

def create_holographic_view(mask, depth_map, sample_rate=0.1):
    """
    Generates an interactive 3D point cloud 'hologram' of the object.
    """
    # Get the coordinates of all pixels that are part of the object's mask
    y_coords, x_coords = np.where(mask > 0)
    
    # If the mask is empty, return an empty figure to avoid errors
    num_points = len(x_coords)
    if num_points == 0:
        return go.Figure()
        
    # To ensure good performance, downsample the points. Use all points if fewer than 10k.
    if num_points > 10000:
        sample_indices = np.random.choice(num_points, int(num_points * sample_rate), replace=False)
    else:
        sample_indices = np.arange(num_points)

    x_sampled = x_coords[sample_indices]
    y_sampled = y_coords[sample_indices]
    
    # Get the depth value for each sampled point. This will be the 'z' axis.
    z_sampled = depth_map[y_sampled, x_sampled]
    
    # Create the 3D scatter plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=x_sampled,
        y=-y_sampled,  # Invert y-axis for correct visual orientation
        z=-z_sampled,  # Invert z-axis so 'deeper' is further into the screen
        mode='markers',
        marker=dict(
            size=2,
            color='cyan',  # Classic hologram color
            opacity=0.8,
            # Optional: add a subtle glow effect
            line=dict(width=0)
        )
    )])
    
    # Apply styling to make it look clean and futuristic
    fig.update_layout(
        scene=dict(
            # Hide axis lines, titles, and tick labels for a clean look
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            # Set a dark background for the scene
            bgcolor="rgba(10, 10, 10, 1)"
        ),
        # Make the plot background transparent to blend with the app
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # Remove any excess margins
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    return fig

