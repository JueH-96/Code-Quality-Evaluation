class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)  # dp[i] = min extra chars from index i to end
        
        for i in range(n - 1, -1, -1):
            # Option 1: consider s[i] as extra character
            dp[i] = 1 + dp[i + 1]
            
            # Option 2: try to match substrings starting at i
            for word in dictionary:
                if i + len(word) <= n and s[i:i+len(word)] == word:
                    dp[i] = min(dp[i], dp[i + len(word)])
        
        return dp[0]