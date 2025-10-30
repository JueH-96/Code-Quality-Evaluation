class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = len(words[i])
        
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i + length - 1
                for k in range(i, j):
                    if words[k][-1] == words[k+1][0]:
                        dp[i][j] = min(dp[i][j], dp[i][k] + len(words[k+1]) - 1)
                    else:
                        dp[i][j] = min(dp[i][j], dp[i][k] + len(words[k+1]))
        
        return dp[0][n-1]