from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def can_reduce_height(time):
            total_reduction = 0
            for workerTime in workerTimes:
                # Calculate the maximum x such that the sum of the first x terms is <= time
                x = int((-1 + (1 + 8 * time / workerTime) ** 0.5) / 2)
                total_reduction += x
            return total_reduction >= mountainHeight

        left, right = 0, mountainHeight * max(workerTimes)
        while left < right:
            mid = (left + right) // 2
            if can_reduce_height(mid):
                right = mid
            else:
                left = mid + 1
        return left