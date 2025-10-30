from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        ordered = sorted(maximumHeight)
        total = 0
        for i, height in enumerate(ordered):
            if i + 1 > height:
                return -1
            total += height
        
        numbers = sorted(list(set(list(range(1, len(maximumHeight)+1)) + maximumHeight)), reverse=True)

        result = 0
        for i, num in enumerate(numbers):
            result += num * i

        return result