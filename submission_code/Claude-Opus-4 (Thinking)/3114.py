class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        # Try each position as the peak
        for peak in range(n):
            heights = [0] * n
            heights[peak] = maxHeights[peak]
            
            # Fill left side (from peak going left)
            for j in range(peak - 1, -1, -1):
                heights[j] = min(maxHeights[j], heights[j + 1])
            
            # Fill right side (from peak going right)
            for j in range(peak + 1, n):
                heights[j] = min(maxHeights[j], heights[j - 1])
            
            # Calculate sum for this configuration
            current_sum = sum(heights)
            max_sum = max(max_sum, current_sum)
        
        return max_sum