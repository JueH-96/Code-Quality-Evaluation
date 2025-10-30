from typing import List
import bisect

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        class BIT:
            def __init__(self, size):
                self.N = size + 2
                self.tree = [0] * self.N

            def update(self, index, value):
                while index < self.N:
                    if value > self.tree[index]:
                        self.tree[index] = value
                    else:
                        break
                    index += index & -index

            def query(self, index):
                res = 0
                while index > 0:
                    if self.tree[index] > res:
                        res = self.tree[index]
                    index -= index & -index
                return res

        n = len(coordinates)
        sorted_coords = sorted(coordinates, key=lambda x: (x[0], x[1]))
        y_vals = sorted(list(set(coord[1] for coord in sorted_coords)))
        y_map = {y: i+1 for i, y in enumerate(y_vals)}  # 1-based indexing for BIT

        # Compute left LIS
        bit = BIT(len(y_vals))
        left = [0] * n
        for i, coord in enumerate(sorted_coords):
            y = coord[1]
            idx = y_map[y]
            max_val = bit.query(idx - 1) + 1
            left[i] = max_val
            bit.update(idx, max_val)

        # Compute right LIS
        sorted_coords_rev = sorted(coordinates, key=lambda x: (-x[0], -x[1]))
        bit = BIT(len(y_vals))
        right = [0] * n
        coord_to_index = {tuple(coord): i for i, coord in enumerate(coordinates)}
        for i, coord in enumerate(sorted_coords_rev):
            y = coord[1]
            idx = y_map[y]
            max_val = bit.query(len(y_vals) - idx) + 1
            right[coord_to_index[tuple(coord)]] = max_val
            bit.update(len(y_vals) - idx + 1, max_val)

        # Find the position of k in sorted_coords
        target = coordinates[k]
        pos = bisect.bisect_left(sorted_coords, target, key=lambda x: (x[0], x[1]))
        return left[pos] + right[k] - 1