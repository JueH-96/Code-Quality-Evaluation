from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        
        # Initialize a 2D table to store the minimum length of str_i
        dp = [[float('inf')] * n for _ in range(n)]
        
        # Initialize the base case
        for i in range(n):
            dp[i][i] = len(words[i])
        
        # Fill the table in a bottom-up manner
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    # Calculate the minimum length of str_j by considering two cases:
                    # 1. join(str_k, words[j])
                    # 2. join(words[j], str_k)
                    if words[k][-1] == words[j][0]:
                        dp[i][j] = min(dp[i][j], dp[i][k] + len(words[j]) - 1)
                    else:
                        dp[i][j] = min(dp[i][j], dp[i][k] + len(words[j]))
                    if words[j][-1] == words[k][0]:
                        dp[i][j] = min(dp[i][j], len(words[j]) + dp[i][k] - 1)
                    else:
                        dp[i][j] = min(dp[i][j], len(words[j]) + dp[i][k])
        
        # The minimum length of str_n-1 is stored in dp[0][n-1]
        return dp[0][n - 1]