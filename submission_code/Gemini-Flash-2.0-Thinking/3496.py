from typing import List
import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def canReduce(time, mountainHeight, workerTimes):
            totalReduction = 0
            for t in workerTimes:
                if t == 0:
                    continue
                # Find the largest h such that t * h * (h + 1) // 2 <= time
                # h * (h + 1) <= 2 * time / t
                y = 2 * time / t
                # Solve h^2 + h - y <= 0
                # h <= (-1 + sqrt(1 + 4 * y)) / 2
                if 1 + 4 * y < 0:
                    maxHeightReduction = 0
                else:
                    maxHeightReduction = int((-1 + math.sqrt(1 + 4 * y)) / 2)
                    if maxHeightReduction < 0:
                        maxHeightReduction = 0
                totalReduction += maxHeightReduction
            return totalReduction >= mountainHeight

        low = 0
        high = mountainHeight * max(workerTimes) if workerTimes else 0
        ans = high + 1

        for t in workerTimes:
            if t == 0 and mountainHeight > 0:
                return -1 # Should not happen based on constraints

        if mountainHeight == 0:
            return 0

        while low <= high:
            mid = (low + high) // 2
            if canReduce(mid, mountainHeight, workerTimes):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        return ans