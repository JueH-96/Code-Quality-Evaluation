class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n)]

        for i in range(n):
            dp[i][0] = 1

        for i in range(1, n):
            for j in range(1, k + 1):
                dp[i][j] = dp[i - 1][j]
                if nums[i] == nums[i - 1]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)
                else:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j] + 1)

        return max(dp[-1])