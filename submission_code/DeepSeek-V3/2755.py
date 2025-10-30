class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n + 1):
            if dp[i] == float('inf'):
                continue
            for word in dictionary:
                l = len(word)
                if i + l <= n and s[i:i+l] == word:
                    dp[i + l] = min(dp[i + l], dp[i])
            dp[i + 1] = min(dp[i + 1], dp[i] + 1)
        return dp[n]