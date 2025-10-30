from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        sorted_max = sorted(maximumHeight, reverse=True)
        n = len(sorted_max)
        total = 0
        if n == 0:
            return 0
        prev = sorted_max[0]
        if prev < 1:
            return -1
        total += prev
        
        for i in range(1, n):
            current_max = sorted_max[i]
            current = min(current_max, prev - 1)
            if current < 1:
                return -1
            total += current
            prev = current
        
        return total