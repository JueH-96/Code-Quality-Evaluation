from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        # Initialize a DP array where dp[i] is the minimum cost to partition nums[0..i]
        dp = [float('inf')] * n
        dp[0] = (nums[0] + k) * cost[0]  # Cost for the first element as a single subarray
        
        for i in range(1, n):
            total_sum = 0
            total_cost = 0
            
            # Try to create subarrays ending at index i
            for j in range(i, -1, -1):
                total_sum += nums[j]
                total_cost += cost[j]
                
                # Calculate the cost of the current subarray nums[j..i]
                subarray_cost = (total_sum + k * (i - j + 1)) * total_cost
                
                # Update the dp[i] with the minimum cost
                if j > 0:
                    dp[i] = min(dp[i], dp[j - 1] + subarray_cost)
                else:
                    dp[i] = min(dp[i], subarray_cost)
        
        return dp[n - 1]