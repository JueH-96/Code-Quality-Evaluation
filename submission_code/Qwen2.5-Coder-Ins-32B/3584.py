from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        m, n = len(word1), len(word2)
        dp = [[-1] * (n + 1) for _ in range(m + 1)]
        
        # Initialize the dp table
        for i in range(m + 1):
            dp[i][0] = i
        
        # Fill the dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = dp[i - 1][j]
        
        # Check if we can form word2
        if dp[m][n] == -1:
            return []
        
        # Reconstruct the sequence
        i, j = m, n
        result = []
        changes = 1
        
        while j > 0:
            if dp[i][j] == dp[i - 1][j - 1]:
                result.append(i - 1)
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] and changes > 0:
                result.append(i - 1)
                i -= 1
                changes -= 1
            else:
                i -= 1
        
        return result[::-1]