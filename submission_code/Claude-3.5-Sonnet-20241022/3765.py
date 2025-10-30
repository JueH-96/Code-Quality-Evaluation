class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate prefix sums for nums and cost arrays
        prefix_nums = [0] * (n + 1)
        prefix_cost = [0] * (n + 1)
        
        for i in range(n):
            prefix_nums[i + 1] = prefix_nums[i] + nums[i]
            prefix_cost[i + 1] = prefix_cost[i] + cost[i]
            
        # dp[i] represents minimum cost to partition nums[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # For each position i
        for i in range(1, n + 1):
            # Try all possible last partitions ending at i
            for j in range(i):
                # Calculate cost for current partition
                partition_sum = prefix_nums[i] - prefix_nums[j]
                partition_cost = prefix_cost[i] - prefix_cost[j]
                
                # Count number of partitions before this one
                partitions_before = 0
                if j > 0:
                    temp = dp[j]
                    if temp != float('inf'):
                        # Count partitions by examining dp value
                        curr_sum = prefix_nums[j]
                        curr_cost = prefix_cost[j]
                        remaining = temp - curr_sum * curr_cost
                        partitions_before = remaining // k if remaining > 0 else 0
                
                # Calculate total cost for current partition
                curr_cost = (partition_sum + k * (partitions_before + 1)) * partition_cost
                
                # Add cost of previous partitions
                total_cost = curr_cost + (dp[j] if j > 0 else 0)
                
                # Update minimum cost
                dp[i] = min(dp[i], total_cost)
        
        return dp[n]