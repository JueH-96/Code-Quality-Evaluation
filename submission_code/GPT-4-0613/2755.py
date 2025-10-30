class Solution:
    def minExtraChar(self, s: str, dictionary: list[str]) -> int:
        n = len(s)
        dp = [0] + [float('inf')] * n
        for i in range(1, n + 1):
            for word in dictionary:
                if s[:i].endswith(word):
                    dp[i] = min(dp[i], dp[i - len(word)])
            dp[i] += 1
        return dp[-1] - 1