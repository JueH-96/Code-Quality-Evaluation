from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        for i in range(n):
            current_heights = [0] * n
            current_heights[i] = maxHeights[i]
            
            # Process left side (from i-1 to 0)
            for j in range(i - 1, -1, -1):
                current_heights[j] = min(maxHeights[j], current_heights[j + 1])
            
            # Process right side (from i+1 to n-1)
            for j in range(i + 1, n):
                current_heights[j] = min(maxHeights[j], current_heights[j - 1])
            
            current_sum = sum(current_heights)
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum