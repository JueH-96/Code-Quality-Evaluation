from typing import List
import math
import itertools

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        best = float('inf')
        # try all possible orders for breaking the locks
        for order in itertools.permutations(strength):
            current_X = 1
            total_time = 0
            valid = True
            for req in order:
                # minimal minutes required such that minutes * current_X >= req
                # minutes = ceil(req / current_X)
                minutes_needed = math.ceil(req / current_X)
                total_time += minutes_needed
                current_X += K  # update factor after breaking the lock
                # early break if already worse than best
                if total_time >= best:
                    valid = False
                    break
            if valid:
                best = min(best, total_time)
        return best

# Example usage:
# sol = Solution()
# print(sol.findMinimumTime([3, 4, 1], 1))  # Output: 4
# print(sol.findMinimumTime([2, 5, 4], 2))  # Output: 5