import numpy as np
from typing import Tuple


class TerrainMap:
    """
    Represents a 2D terrain map with elevation and obstruction data.
    Provides line-of-sight checks and terrain-based movement costs.
    """

    def __init__(self, grid_size: int = 500, seed: int = 42) -> None:
        self.grid_size = grid_size
        rng = np.random.default_rng(seed)
        # Generate elevation map using Perlin-like noise approximation
        base = rng.standard_normal((grid_size // 10, grid_size // 10))
        self.elevation = self._upsample(base, grid_size)
        self.elevation = (self.elevation - self.elevation.min()) / (
            self.elevation.max() - self.elevation.min() + 1e-8
        )
        # Obstruction mask: cells where elevation > 0.8 are obstructed
        self.obstruction_mask = self.elevation > 0.8

    def _upsample(self, array: np.ndarray, target_size: int) -> np.ndarray:
        """Simple bilinear upsampling to target_size x target_size."""
        from scipy.ndimage import zoom

        factor = target_size / array.shape[0]
        return zoom(array, factor, order=1)

    def get_elevation(self, x: float, y: float) -> float:
        """Return terrain elevation at the given (x, y) position."""
        xi = int(np.clip(x, 0, self.grid_size - 1))
        yi = int(np.clip(y, 0, self.grid_size - 1))
        return float(self.elevation[xi, yi])

    def is_obstructed(self, x: float, y: float) -> bool:
        """Return True if position is in an obstructed cell."""
        xi = int(np.clip(x, 0, self.grid_size - 1))
        yi = int(np.clip(y, 0, self.grid_size - 1))
        return bool(self.obstruction_mask[xi, yi])

    def line_of_sight(self, pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> bool:
        """Bresenham line-of-sight check between two 2D points."""
        x0, y0 = int(pos_a[0]), int(pos_a[1])
        x1, y1 = int(pos_b[0]), int(pos_b[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if self.is_obstructed(x0, y0):
                return False
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True

    def get_contour_data(self) -> np.ndarray:
        """Return the full elevation array for visualization."""
        return self.elevation.copy()


if __name__ == "__main__":
    terrain = TerrainMap()
    print(f"TerrainMap created: {terrain.grid_size}x{terrain.grid_size}")
    print(f"Elevation range: [{terrain.elevation.min():.3f}, {terrain.elevation.max():.3f}]")
    print("terrain_map.py OK")
