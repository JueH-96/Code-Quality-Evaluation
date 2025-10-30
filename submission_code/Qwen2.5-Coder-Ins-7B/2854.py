class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[float('inf')] * 26 for _ in range(n)]
        dp[0][ord(words[0][0]) - ord('a')] = len(words[0])
        dp[0][ord(words[0][-1]) - ord('a')] = len(words[0]) - 1
        
        for i in range(1, n):
            x = ord(words[i][0]) - ord('a')
            y = ord(words[i][-1]) - ord('a')
            for j in range(26):
                dp[i][x] = min(dp[i][x], dp[i-1][j] + len(words[i]) - (y == j))
                dp[i][y] = min(dp[i][y], dp[i-1][j] + len(words[i]) - (x == j))
        
        return min(dp[n-1])