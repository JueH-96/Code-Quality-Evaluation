class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i-1] + 1  # Initialize with current character as extra
            for word in dictionary:
                l = len(word)
                if i >= l and s[i-l:i] == word:
                    dp[i] = min(dp[i], dp[i - l])
        return dp[n]