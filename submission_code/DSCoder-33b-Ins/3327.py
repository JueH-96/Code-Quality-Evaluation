class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        dp = [[0] * (n + 1) for _ in range(k + 1)]
        for i in range(1, n + 1):
            dp[0][i] = prefix[i]

        for i in range(1, k + 1):
            for j in range(i, n + 1):
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j - 1] + nums[j - 1])

        ans = float('inf')
        for i in range(n - k + 1):
            ans = min(ans, dp[k][n] - dp[k][i] - prefix[i])

        return ans