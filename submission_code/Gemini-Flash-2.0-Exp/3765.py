class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                num_sum = sum(nums[:i])
                cost_sum = sum(cost[j:i])
                
                dp[i] = min(dp[i], dp[j] + (num_sum + k * (nums[:i].index(nums[i-1]) // (i - j) + 1)) * cost_sum)
        
        return dp[n]