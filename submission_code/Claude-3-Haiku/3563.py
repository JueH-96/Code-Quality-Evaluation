class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * (1 << n) for _ in range(m)]
        
        for i in range(m):
            for mask in range(1 << n):
                if bin(mask).count('1') <= i + 1:
                    for j in range(n):
                        if mask & (1 << j) == 0:
                            dp[i][mask | (1 << j)] = max(dp[i][mask | (1 << j)], dp[i][mask] + grid[i][j])
        
        return max(dp[m-1][mask] for mask in range(1 << n) if bin(mask).count('1') == m)