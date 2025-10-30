class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        if n == 1:
            return maxHeights[0]
        
        # Create arrays to store the maximum heights from left to right and right to left
        left_max = [0] * n
        right_max = [0] * n
        
        # Fill left_max array
        left_max[0] = maxHeights[0]
        for i in range(1, n):
            left_max[i] = min(left_max[i-1], maxHeights[i])
        
        # Fill right_max array
        right_max[n-1] = maxHeights[n-1]
        for i in range(n-2, -1, -1):
            right_max[i] = min(right_max[i+1], maxHeights[i])
        
        # Calculate the maximum sum of heights
        max_sum = 0
        for i in range(n):
            # The height at position i is the minimum of the max heights achievable from left and right
            height_at_i = min(left_max[i], right_max[i])
            max_sum += height_at_i
        
        return max_sum