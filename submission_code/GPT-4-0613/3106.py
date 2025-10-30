class Solution:
    def lengthOfLongestSubsequence(self, nums, target):
        dp = [-1] * (target + 1)
        dp[0] = 0
        for num in nums:
            for j in range(target, num - 1, -1):
                if dp[j - num] != -1:
                    dp[j] = max(dp[j], dp[j - num] + 1)
        return dp[target]