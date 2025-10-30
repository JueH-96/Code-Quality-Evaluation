class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        dp = [[float('-inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for x in range(1, i + 1):
                    if j % 2 == 1:
                        dp[i][j] = max(dp[i][j], dp[x - 1][j - 1] + (prefix_sum[i] - prefix_sum[x - 1]) * (j))
                    else:
                        dp[i][j] = max(dp[i][j], dp[x - 1][j - 1] - (prefix_sum[i] - prefix_sum[x - 1]) * (j))

        return dp[n][k]