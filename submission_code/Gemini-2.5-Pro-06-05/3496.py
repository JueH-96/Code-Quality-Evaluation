from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        
        # This function checks if it's possible to reduce the mountain to height 0
        # within a given time `T`.
        def can_finish(T: int) -> bool:
            total_reduction = 0
            for wt in workerTimes:
                # For each worker, find the maximum height `h` they can reduce.
                # The time to reduce height by `h` is `wt * h * (h+1) / 2`.
                # We need the max `h` such that `wt * h * (h+1) / 2 <= T`.
                # This can be found with a binary search on `h`.
                low_h, high_h = 0, mountainHeight
                max_h_for_worker = 0
                while low_h <= high_h:
                    mid_h = (low_h + high_h) // 2
                    time_needed = wt * mid_h * (mid_h + 1) // 2
                    
                    if time_needed <= T:
                        max_h_for_worker = mid_h
                        low_h = mid_h + 1
                    else:
                        high_h = mid_h - 1
                
                total_reduction += max_h_for_worker
                # Optimization: if total reduction already meets the goal, T is feasible.
                if total_reduction >= mountainHeight:
                    return True
            return False

        # The problem asks for the minimum time, which can be found by binary searching
        # on the time `T`. The `can_finish(T)` function is monotonic.
        
        # Lower bound for time is 0.
        low_T = 0
        # A safe upper bound is the time for the slowest worker to do all the work alone.
        high_T = max(workerTimes) * mountainHeight * (mountainHeight + 1) // 2

        ans = high_T
        
        while low_T <= high_T:
            mid_T = low_T + (high_T - low_T) // 2
            if can_finish(mid_T):
                # `mid_T` is a possible answer, try for a smaller one.
                ans = mid_T
                high_T = mid_T - 1
            else:
                # `mid_T` is too small, need more time.
                low_T = mid_T + 1
                
        return ans