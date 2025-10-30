class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        
        dp[0] = nums[0] * 2 + 1
        prefix_sum = nums[0]
        
        for i in range(1, n):
            dp[i] = (prefix_sum + nums[i]) * ((i+1) & 1) - i - dp[i-1]
            dp[i] = max(dp[i], dp[i-1])
            prefix_sum += nums[i]
            
        return dp[-1]