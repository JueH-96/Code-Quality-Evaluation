from typing import List
import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def time_to_reduce_height(time, height):
            total_time = 0
            for i in range(1, height + 1):
                total_time += time * i
            return total_time
        
        def is_possible(max_time, mountainHeight, workerTimes):
            for time in workerTimes:
                height = 0
                while time_to_reduce_height(time, height + 1) <= max_time:
                    height += 1
                mountainHeight -= height
                if mountainHeight <= 0:
                    return True
            return False
        
        left, right = 1, mountainHeight * max(workerTimes)
        while left < right:
            mid = (left + right) // 2
            if is_possible(mid, mountainHeight, workerTimes):
                right = mid
            else:
                left = mid + 1
        return left