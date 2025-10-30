import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def can_finish(T):
            total = 0
            for time in workerTimes:
                if time > T:
                    continue
                x = int((math.sqrt(1 + 8 * T / time) - 1) // 2)
                total += x
                if total >= mountainHeight:
                    return True
            return total >= mountainHeight

        left, right = 0, 10**18
        while left < right:
            mid = (left + right) // 2
            if can_finish(mid):
                right = mid
            else:
                left = mid + 1
        return left