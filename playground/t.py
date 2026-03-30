import plotly.graph_objects as go
import numpy as np
N=500
N = 200          # liczba punktów
turns = 5        # liczba zwojów
height = 40      # wysokość spirali
radius = 2       # promień spirali

theta = np.linspace(0, 2*np.pi*turns, N)  # kąt
z = np.linspace(0, height, N)            # wysokość
x = radius * np.cos(theta)                # x
y = radius * np.sin(theta)         # y

fig = go.Figure(data=go.Cone(
    x=x,
    y=y,
    z=z,
    u=np.linspace(-1, 1, N),
    v=np.linspace(-1, 1, N),
    w=np.ones(N),
    sizemode="scaled",
    sizeref=2,
    anchor="tip"))

fig.update_layout(
      scene=dict(domain_x=[0, 1],
                 camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

fig.show()