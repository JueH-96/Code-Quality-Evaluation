class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        if len(word2) >= len(word1):
            return []

        n, m = len(word1), len(word2)
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = True

        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = dp[i - 1][j - 1] or (j > 1 and dp[i - 1][j - 2])

        if not dp[n][m]:
            return []

        result = []
        i, j = n, m
        while i > 0 and j > 0:
            if word1[i - 1] == word2[j - 1]:
                result.append(i - 1)
                i -= 1
                j -= 1
            elif j > 1 and dp[i - 1][j - 2]:
                result.append(i - 2)
                i -= 2
                j -= 1
            else:
                i -= 1

        return sorted(result)