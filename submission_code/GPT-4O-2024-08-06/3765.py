class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute prefix sums for nums and cost
        prefix_nums = [0] * (n + 1)
        prefix_cost = [0] * (n + 1)
        
        for i in range(n):
            prefix_nums[i + 1] = prefix_nums[i] + nums[i]
            prefix_cost[i + 1] = prefix_cost[i] + cost[i]
        
        # dp[i] will store the minimum cost to partition nums[0..i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: no elements, no cost
        
        # Iterate over each possible end of subarray
        for i in range(1, n + 1):
            # Try to partition the array ending at i into subarrays
            for j in range(i):
                # Calculate the cost of the subarray nums[j..i-1]
                subarray_sum = prefix_nums[i] - prefix_nums[j]
                subarray_cost = prefix_cost[i] - prefix_cost[j]
                # The order of this subarray is the number of subarrays formed so far + 1
                order = dp[j] // (k + 1) + 1
                current_cost = (subarray_sum + k * order) * subarray_cost
                # Update dp[i] with the minimum cost
                dp[i] = min(dp[i], dp[j] + current_cost)
        
        return dp[n]