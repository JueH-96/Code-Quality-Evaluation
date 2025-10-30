class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        prefix_nums = [0] * (n + 1)
        prefix_cost = [0] * (n + 1)
        
        for i in range(1, n + 1):
            prefix_nums[i] = prefix_nums[i - 1] + nums[i - 1]
            prefix_cost[i] = prefix_cost[i - 1] + cost[i - 1]
        
        # dp[i][j] represents the minimal cost for first i elements split into j subarrays
        # Initialize DP table with infinity
        dp = [[float('inf')] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                for l in range(j - 1, i):
                    if dp[l][j - 1] == float('inf'):
                        continue
                    current_cost = (prefix_nums[i] - prefix_nums[l] + k * j) * (prefix_cost[i] - prefix_cost[l])
                    total_cost = dp[l][j - 1] + current_cost
                    if total_cost < dp[i][j]:
                        dp[i][j] = total_cost
        
        return min(dp[n][1:n + 1])