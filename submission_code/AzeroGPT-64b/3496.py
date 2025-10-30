class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        
        memo = {}

        def helper(h, time):
            if h <= 0:
                return 0

            if (h, time) in memo:
                return memo[(h, time)]

            max_time = 0
            for step in range(1, h + 1):
                cost = time * step + helper(h - step, time)
                max_time = max(max_time, cost)

            memo[(h, time)] = max_time
            return max_time

        max_time = 0

        for time in workerTimes:
            max_time = max(max_time, helper(mountainHeight, time))

        return max_time