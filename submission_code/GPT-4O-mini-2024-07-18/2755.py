class Solution:
    def minExtraChar(self, s: str, dictionary: list[str]) -> int:
        word_set = set(dictionary)
        n = len(s)
        
        # dp[i] will hold the minimum extra characters for s[:i]
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            # Start with the case where we consider the character as extra
            dp[i] = dp[i - 1] + 1
            
            # Check all possible substrings ending at i
            for j in range(i):
                if s[j:i] in word_set:
                    dp[i] = min(dp[i], dp[j])
        
        return dp[n]