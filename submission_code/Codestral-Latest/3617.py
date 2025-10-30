class Solution:
    def possibleStringCount(self, word: str) -> int:
        n = len(word)
        dp = [0] * (n + 1)
        dp[0] = 1

        i = 0
        while i < n:
            dp[i + 1] = dp[i]
            j = i + 1
            while j < n and word[j] == word[i]:
                dp[j + 1] += dp[j]
                j += 1
            i = j

        return dp[n]