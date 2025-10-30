class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        dp = [-1] * (1 << m)
        dp[0] = 0
        for i in range(n):
            next_dp = list(dp)
            for mask in range(1 << m):
                if dp[mask] < 0: continue
                for j in range(m):
                    if ((mask >> j) & 1) == 1: continue
                    next_mask = mask | (1 << j)
                    next_dp[next_mask] = max(next_dp[next_mask], dp[mask] + grid[i][j])
            dp = next_dp
        return max(dp)