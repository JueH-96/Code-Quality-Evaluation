class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # dp[i] represents minimum extra characters for substring s[i:]
        dp = [0] * (n + 1)
        
        # Fill dp array from right to left
        for i in range(n - 1, -1, -1):
            # Option 1: skip current character (count as extra)
            dp[i] = 1 + dp[i + 1]
            
            # Option 2: try to match dictionary words starting at position i
            for word in dictionary:
                word_len = len(word)
                # Check if word can fit and matches
                if i + word_len <= n and s[i:i + word_len] == word:
                    dp[i] = min(dp[i], dp[i + word_len])
        
        return dp[0]