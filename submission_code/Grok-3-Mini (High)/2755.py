class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = 1 + dp[i + 1]
            for word in dictionary:
                word_len = len(word)
                if i + word_len <= n and s[i:i + word_len] == word:
                    dp[i] = min(dp[i], dp[i + word_len])
        return dp[0]