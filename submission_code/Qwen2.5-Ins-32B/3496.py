from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def canReduce(height, time, workers):
            total = 0
            for t in workers:
                x = 1
                while total < height and t * x <= time:
                    total += x
                    x += 1
            return total >= height
        
        left, right = 0, mountainHeight * min(workerTimes) * mountainHeight
        while left < right:
            mid = (left + right) // 2
            if canReduce(mountainHeight, mid, workerTimes):
                right = mid
            else:
                left = mid + 1
        return left