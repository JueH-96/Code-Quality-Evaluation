class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            if nums[i] % 2 == nums[i-1] % 2:
                dp[i] = dp[i-1] + 1
        return max(dp)