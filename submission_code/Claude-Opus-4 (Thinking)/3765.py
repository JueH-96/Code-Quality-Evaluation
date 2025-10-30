class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        
        # Compute prefix sums
        prefix_nums = [0] * (n + 1)
        prefix_cost = [0] * (n + 1)
        for i in range(n):
            prefix_nums[i + 1] = prefix_nums[i] + nums[i]
            prefix_cost[i + 1] = prefix_cost[i] + cost[i]
        
        # DP
        INF = float('inf')
        dp = [[INF] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                for s in range(j - 1, i):
                    # j-th subarray spans indices [s, i-1]
                    subarray_cost = (prefix_nums[i] + k * j) * (prefix_cost[i] - prefix_cost[s])
                    dp[i][j] = min(dp[i][j], dp[s][j - 1] + subarray_cost)
        
        # Find the minimum cost
        return min(dp[n][j] for j in range(1, n + 1))