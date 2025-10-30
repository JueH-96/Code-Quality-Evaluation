class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i - 1], nums[i])
        res = 1
        for i in range(1, n):
            if nums[i] >= dp[i - 1]:
                res += 1
            else:
                dp[i] = dp[i - 1]
        return res