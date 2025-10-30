class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        # Try each index as the peak of the mountain
        for peak in range(n):
            # Create a heights array starting with the peak
            heights = [0] * n
            heights[peak] = maxHeights[peak]
            
            # Fill to the left of the peak
            for i in range(peak - 1, -1, -1):
                heights[i] = min(maxHeights[i], heights[i + 1])
            
            # Fill to the right of the peak
            for i in range(peak + 1, n):
                heights[i] = min(maxHeights[i], heights[i - 1])
            
            # Calculate the sum of this configuration
            current_sum = sum(heights)
            max_sum = max(max_sum, current_sum)
        
        return max_sum