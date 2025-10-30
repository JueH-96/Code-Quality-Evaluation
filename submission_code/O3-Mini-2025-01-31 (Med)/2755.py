from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        wordSet = set(dictionary)
        # dp[i] represents the minimum extra characters from index i to the end
        dp = [0] * (n + 1)
        dp[n] = 0  # base case: no extra characters after the end
        
        # Process from the end of the string backwards
        for i in range(n - 1, -1, -1):
            # Option 1: Mark the current character as extra and move to the next character.
            dp[i] = 1 + dp[i + 1]
            
            # Option 2: Check every possible substring starting from i that is in the dictionary.
            for j in range(i, n):
                # if s[i:j+1] is a valid dictionary word, update dp[i] if choosing it results in fewer extras.
                if s[i:j + 1] in wordSet:
                    dp[i] = min(dp[i], dp[j + 1])
                    
        return dp[0]