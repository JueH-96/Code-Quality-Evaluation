from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # dp[i] = minimum extra chars in s[i:]
        dp = [float('inf')] * (n + 1)
        dp[n] = 0

        # Bottom-up over starting index i
        for i in range(n - 1, -1, -1):
            # Option 1: treat s[i] as an extra char
            dp[i] = 1 + dp[i + 1]
            # Option 2: match any word in dictionary at position i
            for w in dictionary:
                lw = len(w)
                if i + lw <= n and s[i:i+lw] == w:
                    dp[i] = min(dp[i], dp[i+lw])
        return dp[0]