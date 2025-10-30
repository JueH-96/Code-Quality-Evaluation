class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute prefix sums
        prefix_nums = [0] * (n + 1)
        prefix_cost = [0] * (n + 1)
        for i in range(n):
            prefix_nums[i + 1] = prefix_nums[i] + nums[i]
            prefix_cost[i + 1] = prefix_cost[i] + cost[i]
        
        # dp[i][j] = min cost to divide nums[0..i-1] into exactly j subarrays
        INF = float('inf')
        dp = [[INF] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, n) + 1):
                for start in range(j - 1, i):
                    # Last subarray is from start to i-1 (this is the j-th subarray)
                    sum_nums_prefix = prefix_nums[i]  # sum(nums[0..i-1])
                    sum_cost_subarray = prefix_cost[i] - prefix_cost[start]  # sum(cost[start..i-1])
                    subarray_cost = (sum_nums_prefix + k * j) * sum_cost_subarray
                    dp[i][j] = min(dp[i][j], dp[start][j - 1] + subarray_cost)
        
        return min(dp[n][j] for j in range(1, n + 1))