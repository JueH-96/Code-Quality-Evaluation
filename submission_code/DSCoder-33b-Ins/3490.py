class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if (nums[j] + nums[i]) % 2 == 0:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)