from typing import List
from collections import defaultdict

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        # Group coordinates and values by diagonal key d = i - j
        diag_coords = defaultdict(list)
        diag_vals = defaultdict(list)
        for i in range(n):
            for j in range(n):
                d = i - j
                diag_coords[d].append((i, j))
                diag_vals[d].append(grid[i][j])

        # For each diagonal, sort and put back
        for d, coords in diag_coords.items():
            # Ensure coords are in top-left to bottom-right order
            coords.sort(key=lambda x: x[0])
            vals = diag_vals[d]
            # Bottom-left triangle (d >= 0): non-increasing (descending)
            # Top-right triangle (d < 0): non-decreasing (ascending)
            if d >= 0:
                vals.sort(reverse=True)
            else:
                vals.sort()
            # Write sorted values back into grid
            for idx, (i, j) in enumerate(coords):
                grid[i][j] = vals[idx]

        return grid