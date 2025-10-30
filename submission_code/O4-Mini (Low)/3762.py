from typing import List
import math

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        
        # check if we can achieve minimum score K in <= m moves
        def can(K: int) -> bool:
            # need[i] = number of visits required at i
            need = [(K + p - 1) // p for p in points]
            # rightmost index we must reach
            R = max(i for i in range(n) if need[i] > 0)
            # total extra visits beyond the 1st at each
            extra = sum(need[i] - 1 for i in range(n))
            # if no extras, we only need to walk from -1 to R
            if extra == 0:
                return (R + 1) <= m
            
            # Find the extra‐visit position closest to R
            best_d = n
            for i in range(n):
                if need[i] > 1:
                    best_d = min(best_d, abs(R - i))
            
            # baseline: one pass from -1 to R costs (R+1),
            # each extra visit costs 2 moves (go out & back),
            # but for the very last extra we do NOT have to return:
            #   so we save exactly 1 move from those 2
            total_moves = (R + 1) + 2 * extra - 1
            
            # However, if the last extra visit is not at R itself,
            # we must walk from R back to that index and then enter it:
            #   that extra final‐visit cost is (distance + 1) instead of 2,
            #   so we incur delta = (best_d + 1) - 2 = best_d - 1 more moves
            total_moves += (best_d - 1)
            
            return total_moves <= m
        
        # binary search K
        lo, hi = 0, m * max(points) + 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if can(mid):
                lo = mid
            else:
                hi = mid
        return lo