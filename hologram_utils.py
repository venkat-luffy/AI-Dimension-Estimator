# hologram_utils.py

import numpy as np
import plotly.graph_objects as go

def create_holographic_view(mask, depth_map, sample_rate=0.1):
    """
    Generates an interactive 3D point cloud 'hologram' of the object.
    """
    y_coords, x_coords = np.where(mask > 0)
    num_points = len(x_coords)
    if num_points == 0:
        return go.Figure()
    if num_points > 10000:
        sample_indices = np.random.choice(num_points, int(num_points * sample_rate), replace=False)
    else:
        sample_indices = np.arange(num_points)
    x_sampled = x_coords[sample_indices]
    y_sampled = y_coords[sample_indices]
    z_sampled = depth_map[y_sampled, x_sampled]
    fig = go.Figure(data=[go.Scatter3d(
        x=x_sampled,
        y=-y_sampled,
        z=-z_sampled,
        mode='markers',
        marker=dict(
            size=2,
            color='cyan',
            opacity=0.8,
            line=dict(width=0)
        )
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            bgcolor="rgba(10, 10, 10, 1)"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig
