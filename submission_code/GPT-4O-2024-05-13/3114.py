from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        for peak in range(n):
            heights = [0] * n
            heights[peak] = maxHeights[peak]
            
            # Fill left side of the peak
            for i in range(peak - 1, -1, -1):
                heights[i] = min(maxHeights[i], heights[i + 1])
            
            # Fill right side of the peak
            for i in range(peak + 1, n):
                heights[i] = min(maxHeights[i], heights[i - 1])
            
            # Calculate the sum of heights
            current_sum = sum(heights)
            max_sum = max(max_sum, current_sum)
        
        return max_sum