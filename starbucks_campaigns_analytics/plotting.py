import pandas as pd
import plotly_express as px

from plotly.graph_objects import Figure


def funnel_plot(df_to_plot: pd.DataFrame,
                x: str,
                y: str,
                color: str = None) -> Figure:
    """Plots a funnel for `df_to_plot` data
    using `x` as volume, `y` as the steps
    column and `color` to differentiate by
    another column if necessary (optional)
    """
    return px.funnel(df_to_plot, x=x, y=y, color=color)
