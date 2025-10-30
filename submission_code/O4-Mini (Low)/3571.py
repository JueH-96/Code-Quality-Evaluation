from typing import List

class FenwickMax:
    # Fenwick tree for prefix maximum query and point update
    def __init__(self, size: int):
        self.n = size
        self.fw = [0] * (size + 1)
    def update(self, i: int, val: int) -> None:
        # set position i (0-based) to max(current, val)
        i += 1
        while i <= self.n:
            if self.fw[i] < val:
                self.fw[i] = val
            i += i & -i
    def query(self, i: int) -> int:
        # max on prefix [0..i] (0-based)
        if i < 0:
            return 0
        i += 1
        res = 0
        while i > 0:
            if self.fw[i] > res:
                res = self.fw[i]
            i -= i & -i
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        # attach original indices
        pts = [(x, y, idx) for idx, (x, y) in enumerate(coordinates)]
        # compress y
        ys = sorted({y for _, y, _ in pts})
        y_to_i = {y: i for i, y in enumerate(ys)}
        m = len(ys)
        # prepare storage
        dp_end = [0] * n
        dp_start = [0] * n
        
        # 1) LIS ending at each point (forward pass)
        pts_sorted = sorted(pts, key=lambda t: t[0])  # sort by x increasing
        bit = FenwickMax(m)
        i = 0
        while i < n:
            j = i
            # group points with same x: they cannot chain among themselves
            while j < n and pts_sorted[j][0] == pts_sorted[i][0]:
                j += 1
            # compute dp for group [i, j)
            for p in range(i, j):
                x, y, idx = pts_sorted[p]
                yi = y_to_i[y]
                dp_end[idx] = bit.query(yi - 1) + 1
            # update BIT for group
            for p in range(i, j):
                x, y, idx = pts_sorted[p]
                yi = y_to_i[y]
                bit.update(yi, dp_end[idx])
            i = j
        
        # 2) LIS starting at each point (reverse pass)
        # map y to reversed index
        # y_rev = (m-1) - y_to_i[y]
        rev_pts = []
        for x, y, idx in pts:
            yi = y_to_i[y]
            yri = m - 1 - yi
            rev_pts.append((x, yri, idx))
        # sort by x decreasing
        rev_pts.sort(key=lambda t: -t[0])
        bit2 = FenwickMax(m)
        i = 0
        while i < n:
            j = i
            while j < n and rev_pts[j][0] == rev_pts[i][0]:
                j += 1
            for p in range(i, j):
                x, yri, idx = rev_pts[p]
                dp_start[idx] = bit2.query(yri - 1) + 1
            for p in range(i, j):
                x, yri, idx = rev_pts[p]
                bit2.update(yri, dp_start[idx])
            i = j
        
        # The longest path containing k is dp_end[k] + dp_start[k] - 1
        return dp_end[k] + dp_start[k] - 1