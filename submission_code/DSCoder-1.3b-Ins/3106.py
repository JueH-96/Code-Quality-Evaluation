class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [1]*len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j] and nums[i] + dp[j] > dp[i]:
                    dp[i] = dp[j] + 1
        return max(dp)