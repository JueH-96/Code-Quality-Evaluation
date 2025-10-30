from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        # Iterate over all possible peak indices
        for peak_idx in range(n):
            # Initialize heights array with 1s
            heights = [1] * n
            
            # Fill the heights array from the left to the peak
            for i in range(peak_idx):
                heights[i] = min(maxHeights[i], heights[i + 1] + 1)
            
            # Fill the heights array from the right to the peak
            for i in range(n - 1, peak_idx, -1):
                heights[i] = min(maxHeights[i], heights[i - 1] + 1)
            
            # Update the height at the peak
            heights[peak_idx] = min(maxHeights[peak_idx], heights[peak_idx - 1] + 1) if peak_idx > 0 else maxHeights[peak_idx]
            heights[peak_idx] = min(maxHeights[peak_idx], heights[peak_idx + 1] + 1) if peak_idx < n - 1 else maxHeights[peak_idx]
            
            # Update the maximum sum
            max_sum = max(max_sum, sum(heights))
        
        return max_sum