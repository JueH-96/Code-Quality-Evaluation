class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        word_set = set(dictionary)
        
        for i in range(1, n + 1):
            # Option 1: s[i-1] is an extra character
            dp[i] = dp[i-1] + 1
            
            # Option 2: Check if any substring ending at i is in dictionary
            for j in range(i):
                if s[j:i] in word_set:
                    dp[i] = min(dp[i], dp[j])
        
        return dp[n]