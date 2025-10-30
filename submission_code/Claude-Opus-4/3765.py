class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        
        # Compute prefix sums
        prefix_nums = [0]
        prefix_cost = [0]
        
        for i in range(n):
            prefix_nums.append(prefix_nums[-1] + nums[i])
            prefix_cost.append(prefix_cost[-1] + cost[i])
        
        # dp[i][j] = minimum cost to partition nums[0..i-1] into j subarrays
        # We need to track the number of subarrays to calculate the cost correctly
        INF = float('inf')
        dp = [[INF] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for num_subarrays in range(1, i + 1):
                # Try all possible starting positions for the last subarray
                for j in range(num_subarrays - 1, i):
                    if dp[j][num_subarrays - 1] == INF:
                        continue
                    
                    # Cost of subarray from j to i-1
                    # Sum of nums[0..i-1] + k * num_subarrays
                    nums_sum = prefix_nums[i] + k * num_subarrays
                    # Sum of cost[j..i-1]
                    cost_sum = prefix_cost[i] - prefix_cost[j]
                    
                    subarray_cost = nums_sum * cost_sum
                    dp[i][num_subarrays] = min(dp[i][num_subarrays], 
                                               dp[j][num_subarrays - 1] + subarray_cost)
        
        # Find minimum cost among all possible number of subarrays
        result = INF
        for num_subarrays in range(1, n + 1):
            result = min(result, dp[n][num_subarrays])
        
        return result