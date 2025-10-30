class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        coordinates.sort()
        n = len(coordinates)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if coordinates[i][0] > coordinates[j][0] and coordinates[i][1] > coordinates[j][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return dp[k]