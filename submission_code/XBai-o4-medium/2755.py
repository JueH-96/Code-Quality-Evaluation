from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)  # dp[0...n]
        
        for i in range(1, n + 1):
            dp[i] = dp[i-1] + 1  # Initialize with previous + current as extra
            for word in dictionary:
                l = len(word)
                if l <= i:
                    if s[i - l:i] == word:
                        dp[i] = min(dp[i], dp[i - l])
        
        return dp[n]