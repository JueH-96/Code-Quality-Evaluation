from typing import List

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        # Extract and compress y-coordinates
        ys = [y for _, y in coordinates]
        sorted_ys = sorted(set(ys))
        m = len(sorted_ys)
        y_map = {y: i + 1 for i, y in enumerate(sorted_ys)}
        comp_y = [y_map[y] for y in ys]
        # Prepare reversed y for the suffix-LIS computation
        y_rev = [m - cy + 1 for cy in comp_y]
        # Pack points as (x, comp_y, y_rev, original_index)
        pts = [(coordinates[i][0], comp_y[i], y_rev[i], i) for i in range(n)]

        # Fenwick tree / BIT for range maximum queries
        class BIT:
            def __init__(self, size: int):
                self.n = size
                self.tree = [0] * (size + 1)

            def update(self, i: int, val: int) -> None:
                while i <= self.n:
                    if self.tree[i] < val:
                        self.tree[i] = val
                    i += i & -i

            def query(self, i: int) -> int:
                res = 0
                while i > 0:
                    if self.tree[i] > res:
                        res = self.tree[i]
                    i -= i & -i
                return res

        # 1) Compute dp_end[i] = length of longest increasing chain ending at i
        bit1 = BIT(m)
        dp_end = [0] * n
        pts_sorted = sorted(pts, key=lambda p: p[0])  # sort by x ascending
        i = 0
        while i < n:
            j = i
            x_val = pts_sorted[i][0]
            # group by equal x to avoid chaining among same-x points
            while j < n and pts_sorted[j][0] == x_val:
                j += 1
            # compute dp for this group
            for t in range(i, j):
                _, cy, _, idx = pts_sorted[t]
                dp_end[idx] = bit1.query(cy - 1) + 1
            # update BIT after computing the whole group
            for t in range(i, j):
                _, cy, _, idx = pts_sorted[t]
                bit1.update(cy, dp_end[idx])
            i = j

        # 2) Compute dp_start[i] = length of longest increasing chain starting at i
        bit2 = BIT(m)
        dp_start = [0] * n
        pts_sorted_desc = sorted(pts, key=lambda p: -p[0])  # sort by x descending
        i = 0
        while i < n:
            j = i
            x_val = pts_sorted_desc[i][0]
            while j < n and pts_sorted_desc[j][0] == x_val:
                j += 1
            for t in range(i, j):
                _, _, yr, idx = pts_sorted_desc[t]
                dp_start[idx] = bit2.query(yr - 1) + 1
            for t in range(i, j):
                _, _, yr, idx = pts_sorted_desc[t]
                bit2.update(yr, dp_start[idx])
            i = j

        # The answer is the sum of the two LIS values at k, minus 1 (because k is counted twice)
        return dp_end[k] + dp_start[k] - 1