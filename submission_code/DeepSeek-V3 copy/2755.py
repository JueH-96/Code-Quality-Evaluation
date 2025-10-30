class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n + 1):
            if dp[i] == float('inf'):
                continue
            for word in dictionary:
                m = len(word)
                if i + m <= n and s[i:i+m] == word:
                    dp[i + m] = min(dp[i + m], dp[i])
            dp[i + 1] = min(dp[i + 1], dp[i] + 1)
        return dp[n]