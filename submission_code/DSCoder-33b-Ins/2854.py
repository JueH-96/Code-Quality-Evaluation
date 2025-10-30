from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[[float('inf')] * 26 for _ in range(26)] for _ in range(n)]
        dp[0][ord(words[0][0]) - ord('a')][ord(words[0][-1]) - ord('a')] = len(words[0])

        for i in range(1, n):
            for j in range(26):
                for k in range(26):
                    if dp[i - 1][j][k] != float('inf'):
                        dp[i][j][ord(words[i][-1]) - ord('a')] = min(dp[i][j][ord(words[i][-1]) - ord('a')], dp[i - 1][j][k] + len(words[i]) - (k == ord(words[i][0]) - ord('a')))
                        dp[i][ord(words[i][0]) - ord('a')][k] = min(dp[i][ord(words[i][0]) - ord('a')][k], dp[i - 1][j][k] + len(words[i]) - (j == ord(words[i][-1]) - ord('a')))

        return min(min(dp[n - 1][i]) for i in range(26))