class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def time_to_reduce(height, time):
            return sum(time * i for i in range(1, height + 1))

        max_time = 0
        for time in workerTimes:
            height_reduced = 0
            while height_reduced < mountainHeight:
                height_reduced += 1
                max_time = max(max_time, time_to_reduce(height_reduced, time))

        return max_time