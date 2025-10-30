from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[(float('inf'), float('inf')) for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = (words[i][0], words[i][-1])
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    l1, r1 = dp[i][k]
                    l2, r2 = dp[k + 1][j]
                    if r1 == l2:
                        dp[i][j] = min(dp[i][j], (l1, r2), key=lambda x: len(words[k + 1]) - 1 + len(words[i]) - (1 if x[0] == x[1] else 0))
                    else:
                        dp[i][j] = min(dp[i][j], (l1, r2), key=lambda x: len(words[k + 1]) + len(words[i]) - (1 if x[0] == x[1] else 0))
                    if r2 == l1:
                        dp[i][j] = min(dp[i][j], (l2, r1), key=lambda x: len(words[k + 1]) - 1 + len(words[i]) - (1 if x[0] == x[1] else 0))
                    else:
                        dp[i][j] = min(dp[i][j], (l2, r1), key=lambda x: len(words[k + 1]) + len(words[i]) - (1 if x[0] == x[1] else 0))
        
        return sum(len(word) for word in words) - (n - 1) + len(words[0]) - 1 - (1 if dp[0][n - 1][0] == dp[0][n - 1][1] else 0)