from bisect import bisect_left
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, idx, value):
        while idx <= self.n:
            if value > self.tree[idx]:
                self.tree[idx] = value
            else:
                break
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res = max(res, self.tree[idx])
            idx -= idx & -idx
        return res

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = self.n
        self.tree = [0] * (2 * self.n)

    def update(self, idx, value):
        idx += self.n
        if self.tree[idx] >= value:
            return
        self.tree[idx] = value
        while idx > 1:
            idx >>= 1
            new_val = max(self.tree[2 * idx], self.tree[2 * idx + 1])
            if self.tree[idx] == new_val:
                break
            self.tree[idx] = new_val

    def query_range(self, l, r):
        res = 0
        l += self.n
        r += self.n
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Compress y coordinates
        ys = [y for x, y in coordinates]
        sorted_ys = sorted(ys)
        max_rank = len(sorted_ys) - 1

        # Compute in_degree using Fenwick Tree
        sorted_in = sorted(coordinates, key=lambda p: (p[0], -p[1]))
        in_degree = {}
        fenwick_size = len(sorted_ys)
        fenwick_tree = FenwickTree(fenwick_size)
        for point in sorted_in:
            x, y = point
            r = bisect_left(sorted_ys, y)
            max_prev = fenwick_tree.query(r)  # 1-based query up to r (0-based r-1)
            current_in = max_prev + 1
            fenwick_tree.update(r + 1, current_in)  # 1-based index
            in_degree[(x, y)] = current_in

        # Compute out_degree using Segment Tree
        sorted_out = sorted(coordinates, key=lambda p: (-p[0], p[1]))
        out_degree = {}
        segment_size = len(sorted_ys)
        segment_tree = SegmentTree(segment_size)
        for point in sorted_out:
            x, y = point
            r = bisect_left(sorted_ys, y)
            if r + 1 <= max_rank:
                max_out_prev = segment_tree.query_range(r + 1, max_rank)
            else:
                max_out_prev = 0
            current_out = max_out_prev + 1
            segment_tree.update(r, current_out)
            out_degree[(x, y)] = current_out

        # Calculate the result for the k-th point
        x_k, y_k = coordinates[k]
        in_k = in_degree.get((x_k, y_k), 1)
        out_k = out_degree.get((x_k, y_k), 1)
        return in_k + out_k - 1