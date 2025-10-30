class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            dp[i][m] = m
            for j in range(m - 1, -1, -1):
                if word1[i] == word2[j]:
                    dp[i][j] = dp[i + 1][j + 1]
                else:
                    dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j + 1])
        res = []
        i, j = 0, 0
        while i < n and j < m:
            if word1[i] == word2[j]:
                i += 1
                j += 1
            elif dp[i][j] == dp[i + 1][j] + 1:
                res.append(i)
                i += 1
            else:
                res.append(j)
                j += 1
        while i < n:
            res.append(i)
            i += 1
        while j < m:
            res.append(j)
            j += 1
        return res