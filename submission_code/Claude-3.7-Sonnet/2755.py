class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dictionary_set = set(dictionary)  # For O(1) lookups
        
        # dp[i] = min extra characters needed when considering s[0:i]
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            # Default: consider current character as "extra"
            dp[i] = dp[i-1] + 1
            
            # Try to find dictionary words ending at current position
            for j in range(i):
                if s[j:i] in dictionary_set:
                    dp[i] = min(dp[i], dp[j])
        
        return dp[n]