class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        
        # Convert dictionary to set for O(1) lookup
        word_set = set(dictionary)
        
        # For each position i in string
        for i in range(1, n + 1):
            # Default: carry forward previous value + 1 current char as extra
            dp[i] = dp[i-1] + 1
            
            # Try all possible substrings ending at i
            for j in range(i):
                # If substring from j to i exists in dictionary
                if s[j:i] in word_set:
                    dp[i] = min(dp[i], dp[j])
                    
        return dp[n]