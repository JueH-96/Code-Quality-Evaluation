class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n + 1):
                if j - i <= 2 * k:
                    dp[i][j] = max(dp[i][j], dp[i + 1][j])
                    for x in range(i + 1, j):
                        dp[i][j] = max(dp[i][j], dp[i + 1][x] + (nums[i] | nums[x]) * (j - x - 1))
        return dp[0][n]