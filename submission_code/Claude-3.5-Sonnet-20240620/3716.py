class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if i == j + 1:
                    dp[i] = max(dp[i], dp[j] + 1)
                elif abs(nums[i] - nums[j]) <= abs(nums[j] - nums[j-1]):
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)