from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # Initialize the DP table with a large value
        dp = [[float('inf')] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(n + 1):
                # Use the paid painter to paint the current wall
                dp[i][j] = min(dp[i][j], dp[i-1][j] + cost[i-1])
                # Use the free painter to paint the current wall
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1])

        # The result is the minimum cost to paint all walls
        return min(dp[n])

# Example usage:
# cost = [1, 2, 3, 2]
# time = [1, 2, 3, 2]
# solution = Solution()
# print(solution.paintWalls(cost, time))  # Output: 3