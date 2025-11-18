"""AI-generated basic heatmap rendering

originally did -1 to 1, but my values were more like 0 to 2
however, the values i think are wrong (assuming double-moves), so we should go
back to -1 to 1 probably once i fix it

(not doing local normalization because want to be able to compare colors across multiple
outputs)
"""

import numpy as np
from rich.console import Console
from rich.table import Table


def render_heatmap(arr):
    """Render a 3x3 numpy array as a colored heatmap. NaN = not applicable."""
    console = Console()
    table = Table(
        show_header=False,
        show_edge=True,
        pad_edge=True,
        padding=(0, 1),
        box=None,
        show_lines=True,
    )

    # Add columns
    for _ in range(arr.shape[1]):
        table.add_column(justify="center")

    # Color mapping: 0 (red) -> 1 (white) -> 2 (blue)
    def get_style(val):
        if np.isnan(val):
            return "black on dim white", "N/A"

        # Normalize to 0-1 range
        norm = val / 2

        # Interpolate between red (0) and blue (1)
        if norm < 0.5:
            # Red to white
            intensity = int(norm * 2 * 255)
            bg_color = f"rgb(255,{intensity},{intensity})"
        else:
            # White to blue
            intensity = int((1 - norm) * 2 * 255)
            bg_color = f"rgb({intensity},{intensity},255)"

        # Use white text for darker backgrounds, else black
        text_color = "white" if val < 0.5 or val > 1.5 else "black"

        return f"{text_color} on {bg_color}", f"{val:+.2f}"

    # Add rows
    for row in arr:
        cells = []
        for val in row:
            style, text = get_style(val)
            cells.append(f"[{style}]{text}[/{style}]")
        table.add_row(*cells)

    console.print(table)


# Example usage:
arr = np.array([[-1.0, -0.5, 0.0], [0.5, 1.0, np.nan], [-0.3, 0.7, 0.2]])
render_heatmap(arr)
