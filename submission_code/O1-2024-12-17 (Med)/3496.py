class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        import math

        # Helper function to check if we can reduce the mountain of height `mountainHeight`
        # within `timeLimit` seconds using the given workerTimes.
        def canReduceWithin(timeLimit: int) -> bool:
            total_blocks = 0
            for wt in workerTimes:
                # We want x such that wt * x(x+1)/2 <= timeLimit
                # => x(x+1) <= 2 * timeLimit / wt
                # Let A = 2 * timeLimit // wt (integer division).
                # We want x^2 + x <= A.
                A = (2 * timeLimit) // wt
                if A <= 0:
                    continue
                # Solve x^2 + x - A <= 0 for the largest integer x
                # x = floor((-1 + sqrt(1 + 4*A)) / 2)
                x = (math.isqrt(1 + 4 * A) - 1) // 2
                total_blocks += x

                if total_blocks >= mountainHeight:
                    return True
            return False

        # Edge case: if mountainHeight == 0, no time is needed
        if mountainHeight == 0:
            return 0

        # Binary search boundaries:
        # Minimum possible time is 0, but at least 1 second is needed if mountainHeight>0.
        left = 1
        # A safe upper bound is if all the work is done by the slowest worker.
        # That worst-case time is max(workerTimes) * (mountainHeight*(mountainHeight+1)//2).
        right = max(workerTimes) * (mountainHeight * (mountainHeight + 1) // 2)

        # Binary search for the smallest feasible time
        answer = right
        while left <= right:
            mid = (left + right) // 2
            if canReduceWithin(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1

        return answer