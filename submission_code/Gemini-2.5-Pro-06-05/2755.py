from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # Using a set for O(1) average time complexity lookups for words.
        # The time to check `word in word_set` is proportional to len(word) due to hashing.
        word_set = set(dictionary)
        
        # dp[i] will store the minimum number of extra characters for the prefix s[0...i-1].
        # The size is n+1 to handle prefixes of length 0 to n.
        # dp[0] is initialized to 0, representing zero extra characters for an empty prefix.
        dp = [0] * (n + 1)
        
        # Iterate through the string s to compute dp values for each prefix length.
        # i represents the length of the prefix s[:i]. The prefix itself is s[0:i].
        for i in range(1, n + 1):
            # To compute dp[i], we first consider the case where the last character, s[i-1],
            # is an extra character. In this scenario, the cost is 1 (for s[i-1]) plus the
            # minimum cost for the prefix s[:i-1], which is dp[i-1].
            dp[i] = dp[i-1] + 1
            
            # Now, we check if we can achieve a better result. We look for any valid word
            # from the dictionary that ends at index i-1.
            # We iterate through all possible start points 'j' for a substring ending at i-1.
            for j in range(i):
                # The substring under consideration is s[j:i].
                substring = s[j:i]
                
                if substring in word_set:
                    # If s[j:i] is a valid word, we have found a new potential way to form the prefix s[:i].
                    # This way consists of an optimal segmentation of the prefix s[:j], followed by the word s[j:i].
                    # The number of extra characters for this segmentation would be dp[j],
                    # since the word s[j:i] contributes 0 extra characters.
                    # We update dp[i] to the minimum of its current value and this new potential value.
                    dp[i] = min(dp[i], dp[j])
                    
        # After filling the dp table up to n, dp[n] will hold the minimum number of extra
        # characters for the entire string s.
        return dp[n]