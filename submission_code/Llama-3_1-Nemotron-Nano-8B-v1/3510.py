from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        n = len(maxHeight)
        
        # Check if any element is less than its required minimum (i+1)
        for i in range(n):
            if maximumHeight[i] < i + 1:
                return -1
        
        prev = float('inf')
        total = 0
        
        # Process elements in reverse order to maximize the sum
        for i in reversed(range(n)):
            current_max = maximumHeight[i]
            assigned = min(current_max, prev - 1)
            if assigned < 1:
                return -1
            total += assigned
            prev = assigned
        
        return total