class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        m, n = len(word1), len(word2)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[m][n] = True
        
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if word1[i] == word2[j]:
                    dp[i][j] = dp[i + 1][j + 1]
                else:
                    dp[i][j] = dp[i + 1][j] or dp[i][j + 1]
        
        if not dp[0][0]:
            return []
        
        i, j = 0, 0
        result = []
        while i < m and j < n:
            if word1[i] == word2[j]:
                result.append(j)
                i += 1
                j += 1
            elif dp[i + 1][j]:
                i += 1
            else:
                j += 1
        
        return result