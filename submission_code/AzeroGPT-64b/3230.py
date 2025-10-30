class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def distinct(s):
            if not s:
                return True
            p = ord(s[0])
            for i in range(1, len(s)):
                if (ord(s[i]) - p) in (-1, 0, 1):
                    return False
                p = ord(s[i])
            return True

        dp = [[0] * len(word) for _ in range(len(word))]
        for d in range(1, len(word)):
            for i in range(len(word) - d):
                j = i + d
                if distinct(word[i:j+1]):
                    dp[i][j] = 0
                else:
                    dp[i][j] = min(dp[i][k] + dp[k+1][j] for k in range(i, j))

        return dp[0][-1]