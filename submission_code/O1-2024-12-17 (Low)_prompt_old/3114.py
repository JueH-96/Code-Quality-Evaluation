class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        # For each possible peak index i
        for i in range(n):
            # Create an array to store heights
            heights = [0]*n
            
            # Set the peak height as high as possible (its max)
            heights[i] = maxHeights[i]
            
            # Move leftwards from the peak, ensuring non-decreasing from left to right
            for j in range(i-1, -1, -1):
                heights[j] = min(heights[j+1], maxHeights[j])
            
            # Move rightwards from the peak, ensuring non-increasing from left to right
            for j in range(i+1, n):
                heights[j] = min(heights[j-1], maxHeights[j])
            
            # Calculate the sum of this configuration
            cur_sum = sum(heights)
            # Update global maximum sum
            max_sum = max(max_sum, cur_sum)
        
        return max_sum