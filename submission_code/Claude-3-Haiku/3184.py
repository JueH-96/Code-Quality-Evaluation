class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        
        for i in range(1, n):
            max_diff = i
            for j in range(i):
                if nums[i] - nums[j] >= i - j:
                    max_diff = max(max_diff, dp[j] + nums[i])
            dp[i] = max_diff
        
        return max(dp)