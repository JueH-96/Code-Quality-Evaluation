class Solution:
    def semiOrderedPermutation(self, nums):
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[nums[i]] = max(dp[nums[i]], dp[nums[i] + 1] + 1)
        return n - max(dp)