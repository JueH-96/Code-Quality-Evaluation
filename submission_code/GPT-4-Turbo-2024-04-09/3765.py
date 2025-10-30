class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum_nums = [0] * (n + 1)
        prefix_sum_cost = [0] * (n + 1)
        
        # Compute prefix sums for nums and cost
        for i in range(n):
            prefix_sum_nums[i + 1] = prefix_sum_nums[i] + nums[i]
            prefix_sum_cost[i + 1] = prefix_sum_cost[i] + cost[i]
        
        # dp[i] will hold the minimum cost to partition the array up to index i
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Compute the minimum cost for each subarray ending at position j
        for j in range(1, n + 1):
            for i in range(j):
                sum_nums = prefix_sum_nums[j] - prefix_sum_nums[i]
                sum_cost = prefix_sum_cost[j] - prefix_sum_cost[i]
                subarray_count = len(dp[:i])
                subarray_cost = (sum_nums + k * subarray_count) * sum_cost
                dp[j] = min(dp[j], dp[i] + subarray_cost)
        
        return dp[n]