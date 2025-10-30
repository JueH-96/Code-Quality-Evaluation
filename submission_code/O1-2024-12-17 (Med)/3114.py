class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        answer = 0
        
        # For each possible peak index i
        for i in range(n):
            heights = [0] * n
            
            # Set the peak tower's height to the maximum allowed
            heights[i] = maxHeights[i]
            
            # Move left from i, ensuring non-decreasing towards the peak
            for j in range(i - 1, -1, -1):
                heights[j] = min(heights[j + 1], maxHeights[j])
            
            # Move right from i, ensuring non-increasing away from the peak
            for j in range(i + 1, n):
                heights[j] = min(heights[j - 1], maxHeights[j])
            
            # Calculate the sum of this configuration
            current_sum = sum(heights)
            answer = max(answer, current_sum)
        
        return answer