class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        dp = [[1e9] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(min(m, i)):
                if nums[i - 1] & andValues[j] == andValues[j]:
                    dp[i][j + 1] = min(dp[i][j + 1], dp[i - 1][j] + nums[i - 1])

        return dp[n][m] if dp[n][m] < 1e9 else -1