import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_n_rows():
    n = len(wspolczynniki)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"a={p['a']}, b={p['b']}" for p in wspolczynniki]
    )

    for i, p in enumerate(wspolczynniki, start=1):
        a, b = p["a"], p["b"]
        y = a * np.sin(x) + b  # tu wstaw swój model/zależność

        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=f"wariant {i}"),
            row=i, col=1
        )

    fig.update_layout(
        height=250 * n,  # ważne, żeby wszystkie wiersze były czytelne
        title="Zmiana danych dla różnych współczynników",
        showlegend=False
    )

    fig.show()
