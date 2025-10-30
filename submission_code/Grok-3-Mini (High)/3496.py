from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def max_reduction(w: int, T: int) -> int:
            left, right = 0, 300000000  # A large upper bound to cover maximum possible x
            res = 0
            while left <= right:
                mid = (left + right) // 2
                val = w * (mid * (mid + 1) // 2)
                if val <= T:
                    res = mid
                    left = mid + 1
                else:
                    right = mid - 1
            return res
        
        def feasible(T: int) -> bool:
            total = 0
            for w in workerTimes:
                max_red = max_reduction(w, T)
                total += max_red
                if total >= mountainHeight:
                    return True
            return total >= mountainHeight
        
        # Binary search for the minimum time
        max_w = max(workerTimes)
        max_T = max_w * mountainHeight * (mountainHeight + 1) // 2
        left, right = 0, max_T
        while left <= right:
            mid = (left + right) // 2
            if feasible(mid):
                right = mid - 1
            else:
                left = mid + 1
        # After binary search, left is the smallest T where feasible is true
        return left