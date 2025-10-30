class Solution:
    def minimizeConcatenatedLength(self, words):
        n = len(words)
        dp = [[0]*n for _ in range(n)]
        words = sorted(words, key=lambda x: (x[0], x[-1]))
        for i in range(n):
            dp[i][i] = len(words[i])
        for l in range(1, n):
            for i in range(n-l):
                j = i + l
                dp[i][j] = min(dp[i+1][j] + len(words[i]) - (words[i][-1] == words[i+1][0]), dp[i][j-1] + len(words[j]) - (words[i][0] == words[j][-1]))
        return dp[0][n-1]