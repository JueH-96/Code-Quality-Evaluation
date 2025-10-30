from typing import List

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Sort the coordinates based on x first, then y
        coordinates.sort()
        
        # Extract the target point
        target = coordinates[k]
        
        # Initialize the dp array to store the maximum path length ending at each point
        dp = [1] * len(coordinates)
        
        # Initialize the result with 1 as the minimum length of the path
        result = 1
        
        # Iterate through the sorted coordinates
        for i in range(len(coordinates)):
            # Check if the current point can be reached from the target point
            if coordinates[i][0] > target[0] and coordinates[i][1] > target[1]:
                # Iterate through all previous points to find the longest path
                for j in range(i):
                    if coordinates[j][0] < coordinates[i][0] and coordinates[j][1] < coordinates[i][1]:
                        dp[i] = max(dp[i], dp[j] + 1)
                # Update the result with the maximum path length
                result = max(result, dp[i])
        
        # Return the result
        return result

# Example usage:
# sol = Solution()
# print(sol.maxPathLength([[3,1],[2,2],[4,1],[0,0],[5,3]], 1)) # Output: 3
# print(sol.maxPathLength([[2,1],[7,0],[5,6]], 2)) # Output: 2