from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort(reverse=True)
        total = 0
        current = float('inf')
        for h in maximumHeight:
            allowed = min(h, current - 1)
            if allowed <= 0:
                return -1
            total += allowed
            current = allowed
        return total