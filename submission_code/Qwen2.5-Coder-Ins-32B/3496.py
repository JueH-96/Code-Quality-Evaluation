from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def canFinishInTime(time_limit: int) -> bool:
            total_height_reduced = 0
            for worker_time in workerTimes:
                # Calculate the maximum height this worker can reduce within time_limit
                x = time_limit // worker_time
                total_height_reduced += x * (x + 1) // 2
                if total_height_reduced >= mountainHeight:
                    return True
            return False

        left, right = 1, mountainHeight * max(workerTimes)
        while left < right:
            mid = (left + right) // 2
            if canFinishInTime(mid):
                right = mid
            else:
                left = mid + 1
        return left