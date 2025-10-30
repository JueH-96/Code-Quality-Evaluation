from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0  # No extra characters needed at the end

        for i in range(n - 1, -1, -1):
            # Option 1: Skip the current character
            dp[i] = dp[i + 1] + 1

            # Option 2: Try to match any word starting at position i
            for word in dictionary:
                end = i + len(word)
                if end <= n and s[i:end] == word:
                    dp[i] = min(dp[i], dp[end])

        return dp[0]