class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = dp[i-1] + nums[i]
            if nums[i] % 2 != nums[i-1] % 2:
                dp[i] -= x
        return max(dp)