import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        workerTimes.sort()  # Sort to process fastest workers first for early exit

        def possible(T: int) -> bool:
            sum_x = 0
            for wt in workerTimes:
                if wt > T:
                    continue  # Worker can't contribute any work in time T
                # Binary search to find the maximum x where x(x+1) <= 2*T/wt
                S = 2 * T / wt
                low = 0
                high = int(math.sqrt(S)) + 1
                best = 0
                while low <= high:
                    mid = (low + high) // 2
                    product = mid * (mid + 1)
                    if product <= S:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                sum_x += best
                if sum_x >= mountainHeight:
                    return True
            return sum_x >= mountainHeight

        # Determine upper bound using the fastest worker
        fastest = workerTimes[0]
        upper_bound = fastest * mountainHeight * (mountainHeight + 1) // 2

        # Binary search on time T
        left, right = 0, upper_bound
        while left < right:
            mid = (left + right) // 2
            if possible(mid):
                right = mid
            else:
                left = mid + 1
        return left