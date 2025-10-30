class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n = len(grid)
        dp = [0] * (k + 1)
        
        for i in range(n):
            row = sorted(grid[i], reverse=True)
            limit = limits[i]
            new_dp = dp[:]
            
            for j in range(min(k, limit)):
                for x in range(k - j):
                    new_dp[x + j + 1] = max(new_dp[x + j + 1], dp[x] + row[j])
            
            dp = new_dp
        
        return max(dp)