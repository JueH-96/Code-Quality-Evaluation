class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        dp = [[0] * (n + 1) for _ in range(4)]
        for i in range(1, 4):
            for j in range(i * 2 - 1, n + 1):
                if i == 1:
                    dp[i][j] = prefix_sum[j]
                else:
                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j - 2] + nums[j - 1])

        return dp[3][n]