class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0]*n
        dp2 = [0]*n
        dp[0] = nums[0]
        dp2[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i-1]+nums[i])
            dp2[i] = max(dp2[i-1]+nums[i], nums[i])
        return max(max(dp), max(dp2))