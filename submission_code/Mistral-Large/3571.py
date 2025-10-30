from typing import List
import bisect

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Sort coordinates by x and then by y
        coordinates.sort()

        # Initialize DP array
        dp = [1] * len(coordinates)

        # Iterate through the sorted coordinates
        for i in range(len(coordinates)):
            xi, yi = coordinates[i]
            for j in range(i):
                xj, yj = coordinates[j]
                if xj < xi and yj < yi:
                    dp[i] = max(dp[i], dp[j] + 1)

        # Find the maximum length of the increasing path that includes coordinates[k]
        original_x, original_y = coordinates[k]
        max_length = 0

        for i in range(len(coordinates)):
            xi, yi = coordinates[i]
            if xi == original_x and yi == original_y:
                max_length = max(max_length, dp[i])

        return max_length

# Example usage:
# coordinates = [[3,1],[2,2],[4,1],[0,0],[5,3]]
# k = 1
# solution = Solution()
# print(solution.maxPathLength(coordinates, k))  # Output: 3