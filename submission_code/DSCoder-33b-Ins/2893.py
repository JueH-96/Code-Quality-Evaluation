class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        max_score = nums[0]

        for i in range(1, n):
            if nums[i] % 2 == nums[i - 1] % 2:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = max(dp[i - 1] + nums[i] - x, nums[i])
            max_score = max(max_score, dp[i])

        return max_score