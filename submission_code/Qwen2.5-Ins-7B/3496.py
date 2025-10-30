from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        workerTimes.sort()
        total_time = 0
        for i, time in enumerate(workerTimes):
            required_time = 0
            x = 1
            while required_time < mountainHeight:
                required_time += time * x
                x += 1
            total_time = max(total_time, required_time - time)
        return total_time