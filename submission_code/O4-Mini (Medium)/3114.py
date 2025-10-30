from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        best = 0
        
        # Try each index i as the peak
        for peak in range(n):
            # Build heights array for this peak
            heights = [0] * n
            # Set the peak height to its maximum
            heights[peak] = maxHeights[peak]
            
            # Fill to the left of the peak: non-decreasing toward the peak
            for i in range(peak - 1, -1, -1):
                # Can't exceed its own max or the next height
                heights[i] = min(maxHeights[i], heights[i + 1])
            
            # Fill to the right of the peak: non-increasing away from the peak
            for i in range(peak + 1, n):
                heights[i] = min(maxHeights[i], heights[i - 1])
            
            # Sum this configuration
            total = sum(heights)
            if total > best:
                best = total
        
        return best