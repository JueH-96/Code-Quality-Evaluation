class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        
        for i in range(n):
            dp[i] = nums[i]  # Start with the value itself
            for j in range(i):
                if nums[i] - nums[j] >= i - j:  # Check the balanced condition
                    dp[i] = max(dp[i], dp[j] + nums[i])  # Update the maximum sum
        
        return max(dp)  # Return the maximum sum found