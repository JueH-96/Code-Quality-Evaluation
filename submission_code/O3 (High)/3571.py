from typing import List
from collections import defaultdict

class Solution:
    # ---------- internal Fenwick (Binary Indexed) Tree ----------
    class _BIT:
        def __init__(self, size: int):
            self.N = size + 2                      # 1-based indexing
            self.bit = [0] * (self.N)

        def update(self, idx: int, val: int):
            """Set position idx to max(current, val)."""
            while idx < self.N:
                if val > self.bit[idx]:
                    self.bit[idx] = val
                idx += idx & -idx

        def query(self, idx: int) -> int:
            """Maximum value on prefix [1 .. idx]."""
            res = 0
            while idx:
                if self.bit[idx] > res:
                    res = self.bit[idx]
                idx -= idx & -idx
            return res
    # ----------------------------------------------------------------

    def _longest_chain_ending_at_each_point(self, xs: List[int], ys: List[int],
                                            order_sign: int) -> List[int]:
        """
        order_sign =  1  -> chain with x, y strictly increasing
        order_sign = -1  -> chain with x, y strictly decreasing
        Returns dp[i] = length of the longest chain that ENDS at point i
                        w.r.t the selected order.
        """

        n = len(xs)
        # transform according to desired order
        tx  = [order_sign * x for x in xs]
        ty  = [order_sign * y for y in ys]

        # coordinate–compress ty
        uniq_ty = sorted(set(ty))
        comp = {v: i + 1 for i, v in enumerate(uniq_ty)}  # 1-based

        # prepare list of points with original indices
        pts = [(tx[i], ty[i], i) for i in range(n)]
        pts.sort(key=lambda p: (p[0], p[1]))              # sort by transformed x then y

        bit = self._BIT(len(uniq_ty))
        dp  = [0] * n

        i = 0
        while i < n:
            j = i
            # gather all points that share identical transformed x  ( = identical real x )
            while j < n and pts[j][0] == pts[i][0]:
                j += 1

            # first compute dp for the whole group
            tmp = []
            for t in range(i, j):
                _, y_val, idx_orig = pts[t]
                y_idx = comp[y_val]
                best  = bit.query(y_idx - 1)               # y must be strictly smaller
                dp[idx_orig] = best + 1
                tmp.append((y_idx, dp[idx_orig]))

            # THEN update BIT (so points with identical x don't influence each other)
            for y_idx, value in tmp:
                bit.update(y_idx, value)

            i = j
        return dp

    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        xs = [p[0] for p in coordinates]
        ys = [p[1] for p in coordinates]

        # longest chain that ends at each point (x,y increasing)
        forward = self._longest_chain_ending_at_each_point(xs, ys, order_sign=1)

        # longest chain that ends at each point when we look at the plane "up-side-down"
        # (i.e. x, y strictly DECREASING in original coordinates)
        backward = self._longest_chain_ending_at_each_point(xs, ys, order_sign=-1)

        # chain that CONTAINS point k = longest prefix ending at k  + longest suffix starting at k - 1
        return forward[k] + backward[k] - 1