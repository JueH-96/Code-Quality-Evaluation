from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        max_heights = sorted(set(maximumHeight), reverse=True)
        
        if len(max_heights) < n:
            return -1
        
        assigned_heights = [0] * n
        height_index = 0
        
        for i in range(n):
            while height_index < len(max_heights) and max_heights[height_index] in assigned_heights:
                height_index += 1
            if height_index == len(max_heights):
                return -1
            assigned_heights[i] = max_heights[height_index]
            height_index += 1
        
        return sum(assigned_heights)