class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = dp[i]
            cur = 0
            for j in range(i, -1, -1):
                cur += nums[j] * (1 if (i - j) % 2 == 0 else -1)
                dp[i + 1] = max(dp[i + 1], dp[j] + abs(cur))
        return dp[n]