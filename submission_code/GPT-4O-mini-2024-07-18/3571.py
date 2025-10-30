from typing import List

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Extract the target point
        target = coordinates[k]
        
        # Sort coordinates based on x and then y
        coordinates.sort()
        
        # Dynamic programming array to store the maximum path length ending at each point
        dp = [1] * len(coordinates)
        
        # Map to store the maximum path length for each point
        max_length = 0
        
        for i in range(len(coordinates)):
            x_i, y_i = coordinates[i]
            # If the current point is the target point, we want to track its maximum path length
            if (x_i, y_i) == (target[0], target[1]):
                max_length = dp[i]
            
            for j in range(i):
                x_j, y_j = coordinates[j]
                if x_j < x_i and y_j < y_i:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        # The result is the maximum length found that includes the target point
        return max_length