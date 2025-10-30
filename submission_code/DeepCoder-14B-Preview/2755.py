import math

class Solution:
    def minExtraChar(self, s: str, dictionary: list) -> int:
        word_set = set(dictionary)
        max_len = max(len(word) for word in word_set) if word_set else 0
        n = len(s)
        INF = math.inf
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for i in range(n + 1):
            if dp[i] == INF:
                continue
            # Try all possible substrings starting at i
            for l in range(1, max_len + 1):
                if i + l > n:
                    continue
                substr = s[i:i+l]
                if substr in word_set:
                    if dp[i + l] > dp[i]:
                        dp[i + l] = dp[i]
            # Consider not taking any substring, adding 1 to the extra and moving to i+1
            if i < n:
                if dp[i + 1] > dp[i] + 1:
                    dp[i + 1] = dp[i] + 1
        
        min_extra = min(dp[j] + (n - j) for j in range(n + 1))
        return min_extra