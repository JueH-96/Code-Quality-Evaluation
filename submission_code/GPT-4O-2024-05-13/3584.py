class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        if m >= n:
            return []
        
        # dp[i][j] will be True if word2[:j] can be formed using word1[:i] with at most one change
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = True
        
        for i in range(1, n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = dp[i - 1][j - 1] or dp[i - 1][j]
        
        if not dp[n][m]:
            return []
        
        result = []
        i, j = n, m
        while j > 0:
            if dp[i - 1][j]:
                i -= 1
            else:
                result.append(i - 1)
                i -= 1
                j -= 1
        
        result.reverse()
        return result