class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0 for _ in range(2 * k + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, min(i + 1, 2 * k + 1)):
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] | nums[i - 1])
        res = 0
        for i in range(k, 2 * k + 1):
            for j in range(i - k, k + 1):
                res = max(res, dp[i][j] ^ dp[i][2 * k - j])
        return res