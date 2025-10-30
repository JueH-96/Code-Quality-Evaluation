class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        if not nums:
            return 0

        n = len(nums)
        dp = [1] * n  # dp[i] will store the length of the longest subsequence ending at index i

        for i in range(1, n):
            for j in range(i):
                if abs(nums[i] - nums[j]) <= abs(nums[i-1] - nums[i-2]):
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)