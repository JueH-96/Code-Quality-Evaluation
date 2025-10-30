class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        from math import isqrt

        workerTimes.sort()
        n = len(workerTimes)

        def isPossible(T):
            total_work = 0
            for k in workerTimes:
                S = 2 * T // k
                D = 1 + 4 * S
                sqrt_D = isqrt(D)
                x_max = (sqrt_D - 1) // 2
                x_max = max(x_max, 0)
                total_work += x_max
            return total_work >= mountainHeight

        left = 0
        right = 1
        while not isPossible(right):
            right *= 2

        while left < right:
            mid = (left + right) // 2
            if isPossible(mid):
                right = mid
            else:
                left = mid + 1
        return left