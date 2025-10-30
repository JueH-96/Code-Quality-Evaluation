from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 0
        
        # Convert dictionary to a set for quick look-up
        word_set = set(dictionary)
        # Precompute the lengths of the words in the dictionary
        word_lengths = set(len(word) for word in dictionary)
        
        for i in range(1, n + 1):
            # Assume the current character is extra
            dp[i] = dp[i - 1] + 1
            
            # Check all possible word lengths ending at position i
            for length in word_lengths:
                if length <= i and s[i - length:i] in word_set:
                    dp[i] = min(dp[i], dp[i - length])
        
        return dp[n]