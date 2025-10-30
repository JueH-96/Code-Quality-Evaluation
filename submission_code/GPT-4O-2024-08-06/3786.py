class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        def canTransform(a, b, k):
            diff = abs(ord(a) - ord(b))
            return min(diff, 26 - diff) <= k

        n = len(s)
        dp = [[0] * n for _ in range(n)]

        for i in range(n):
            dp[i][i] = 1

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                elif canTransform(s[i], s[j], k):
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

        return dp[0][n - 1]