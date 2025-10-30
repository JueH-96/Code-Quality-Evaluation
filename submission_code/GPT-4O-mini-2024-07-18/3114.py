from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        # Iterate over each possible peak
        for peak in range(n):
            heights = [0] * n
            
            # Set the peak height
            heights[peak] = maxHeights[peak]
            
            # Fill heights to the left of the peak
            for i in range(peak - 1, -1, -1):
                heights[i] = min(heights[i + 1], maxHeights[i])
            
            # Fill heights to the right of the peak
            for i in range(peak + 1, n):
                heights[i] = min(heights[i - 1], maxHeights[i])
            
            # Calculate the sum of the current configuration
            current_sum = sum(heights)
            max_sum = max(max_sum, current_sum)
        
        return max_sum