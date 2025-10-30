from math import ceil
from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        # if we can't even visit each index once, answer is 0.
        if m < n:
            return 0
        
        # check feasibility for candidate score x
        def feasible(x: int) -> bool:
            extra0 = extraN = 0
            extra_mid = 0
            for i in range(n):
                # required visits for index i
                req = (x + points[i] - 1) // points[i]
                # base visit is already present in the forward path.
                extra = req - 1
                if extra < 0:
                    extra = 0
                if i == 0:
                    extra0 = extra
                elif i == n - 1:
                    extraN = extra
                else:
                    extra_mid += extra
            # extra moves required:
            # endpoints: each extra costs 1 move,
            # middle positions: each extra costs 2 moves.
            additional_moves = extra0 + extraN + 2 * extra_mid
            
            # total moves = base visits (n moves) + additional bounce moves
            return n + additional_moves <= m
        
        # binary search on answer x. Lower bound 0, high bound:
        lo, hi = 0, m * max(points)
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans