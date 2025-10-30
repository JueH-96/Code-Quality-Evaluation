import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # We want to find the smallest T such that we can distribute
        # the work (which sums up to mountainHeight units) among all workers
        # without exceeding T seconds for any worker.

        # Time taken by worker i to remove x units is:
        #   workerTimes[i] * (1 + 2 + ... + x) = workerTimes[i] * (x * (x + 1) // 2)
        #
        # We'll use a binary search for T over the range [0, max_possible],
        # where max_possible = max(workerTimes) * (mountainHeight * (mountainHeight + 1) // 2).
        #
        # For a given T, we find the maximum x_i each worker i can remove
        # by solving:
        #   workerTimes[i] * (x_i * (x_i + 1) // 2) <= T
        # Then sum these maximum x_i for all workers. If the sum >= mountainHeight,
        # it means we can finish in T seconds (or less), otherwise we cannot.

        # Edge case: if mountainHeight == 0 (though per constraints it's at least 1)
        # we would return 0.

        # Upper bound for the binary search
        max_w = max(workerTimes)
        # max units cost if only this slowest worker does all the work
        right = max_w * (mountainHeight * (mountainHeight + 1) // 2)
        left = 0

        # Function to check how many units can be removed within time t
        def units_removed_in_time(t: int) -> int:
            total_units = 0
            for w in workerTimes:
                if w == 0:
                    continue
                # We want the largest x such that w * (x * (x + 1) // 2) <= t
                # => x * (x + 1) <= 2 * t / w
                # Let M = 2 * (t // w) (integer division)
                # Then x^2 + x <= M
                # Solve x = floor( (sqrt(1 + 4M) - 1) / 2 )
                M = 2 * (t // w)
                # use integer square root to avoid floating imprecision
                x = (math.isqrt(1 + 4 * M) - 1) // 2
                total_units += x
                # Early break if we've already removed enough
                if total_units >= mountainHeight:
                    break
            return total_units

        # Binary search
        while left < right:
            mid = (left + right) // 2
            if units_removed_in_time(mid) >= mountainHeight:
                right = mid
            else:
                left = mid + 1

        return left