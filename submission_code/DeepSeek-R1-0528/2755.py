class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = 1 + dp[i + 1]
            for word in dictionary:
                len_word = len(word)
                if i + len_word <= n and s[i:i+len_word] == word:
                    dp[i] = min(dp[i], dp[i + len_word])
        return dp[0]