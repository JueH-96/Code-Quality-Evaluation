class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        # Convert the dictionary into a set for faster lookups
        dict_set = set(dictionary)
        n = len(s)
        
        # dp[i] will hold the minimum number of extra characters needed for s[:i]
        dp = [0] * (n+1)
        
        for i in range(1, n+1):
            # By default, consider the character at i-1 as extra and build from dp[i-1]
            dp[i] = dp[i-1] + 1
            
            # Try to match any dictionary word ending at position i-1
            for w in dict_set:
                if len(w) <= i and s[i-len(w):i] == w:
                    dp[i] = min(dp[i], dp[i-len(w)])
        
        return dp[n]