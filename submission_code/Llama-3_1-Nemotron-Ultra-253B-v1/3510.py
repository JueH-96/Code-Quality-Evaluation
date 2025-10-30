from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        prev = float('inf')
        total = 0
        for h in reversed(maximumHeight):
            current = min(h, prev - 1)
            if current <= 0:
                return -1
            total += current
            prev = current
        return total