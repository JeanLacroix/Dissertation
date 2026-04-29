"""Shared colour constants and colormaps for model visualisations."""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

INK = "#203128"
MUTED = "#5D6F64"
GRID = "#D5E2D9"
NEUTRAL = "#7F8A82"

PRIMARY = "#1F6B4F"
SECONDARY = "#2F8A64"
TERTIARY = "#56A07C"
ACCENT = "#7CA35A"
LIGHT = "#9FCFB0"
PALE = "#DCEFE2"
DARK = "#154734"


def green_cmap(name: str = "company_green") -> LinearSegmentedColormap:
    """Green cmap."""
    return LinearSegmentedColormap.from_list(name, [PALE, LIGHT, SECONDARY, PRIMARY, DARK])


def green_soft_cmap(name: str = "company_green_soft") -> LinearSegmentedColormap:
    """Green soft cmap."""
    return LinearSegmentedColormap.from_list(name, ["#F3F8F4", PALE, LIGHT, SECONDARY, PRIMARY])
