class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            if nums[i] <= nums[i-1]:
                dp[i] = nums[i] + dp[i-1]
            else:
                dp[i] = nums[i]
        return max(dp)