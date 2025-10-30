class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n, m = len(grid), len(grid[0])
        dp = [[0] * (k + 1) for _ in range(n)]
        
        for i in range(n):
            grid[i].sort(reverse=True)
            for j in range(min(limits[i], k + 1)):
                dp[i][j] = grid[i][j]
        
        for i in range(1, n):
            for j in range(1, k + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1] + grid[i][j - 1])
        
        return dp[n - 1][k]