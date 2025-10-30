class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # Convert dictionary to set for O(1) lookup
        word_set = set(dictionary)
        
        # dp[i] represents the minimum extra characters from index i to end of string
        dp = [0] * (n + 1)
        
        # Build dp array from right to left
        for i in range(n - 1, -1, -1):
            # Option 1: Skip current character (count as extra)
            dp[i] = dp[i + 1] + 1
            
            # Option 2: Try all possible substrings starting at position i
            for j in range(i + 1, n + 1):
                substring = s[i:j]
                if substring in word_set:
                    # If substring is in dictionary, we can use it
                    # No extra characters for this substring
                    dp[i] = min(dp[i], dp[j])
        
        return dp[0]