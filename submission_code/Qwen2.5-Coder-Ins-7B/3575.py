class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        for i in range(1, k + 1):
            for j in range(i, n + 1):
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j - i] | nums[j - 1])
        return dp[k][n]