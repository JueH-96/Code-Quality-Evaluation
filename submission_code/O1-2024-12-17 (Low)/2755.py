class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dict_set = set(dictionary)
        
        # dp[i] will store the minimum number of extra characters
        # required to break s[:i] optimally.
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            # Option 1: skip s[i-1], so dp[i] = dp[i-1] + 1
            dp[i] = dp[i-1] + 1
            
            # Option 2: use a word in the dictionary if it matches s[j:i]
            for j in range(i):
                if s[j:i] in dict_set:
                    dp[i] = min(dp[i], dp[j])
                    
        return dp[n]