from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i-1]
            for word in dictionary:
                l = len(word)
                if l > i:
                    continue
                if s[i-l:i] == word:
                    if dp[i - l] + l > dp[i]:
                        dp[i] = dp[i - l] + l
        return len(s) - dp[n]