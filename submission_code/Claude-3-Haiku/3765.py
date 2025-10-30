class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for l in range(i):
                    total_cost = (prefix_sum[i] - prefix_sum[l]) * (sum(cost[l:i]))
                    dp[i][j] = min(dp[i][j], dp[l][j - 1] + total_cost)
        
        return dp[n][k]