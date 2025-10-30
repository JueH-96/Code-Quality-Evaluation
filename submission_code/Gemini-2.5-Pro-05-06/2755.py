from typing import List

class Solution:
  def minExtraChar(self, s: str, dictionary: List[str]) -> int:
    n = len(s)
    dictionary_set = set(dictionary)
    
    # dp[k] stores the minimum extra characters for the prefix s[0...k-1] (which has length k).
    # dp is 1-indexed effectively for string length, so dp[0] is for empty string, dp[n] for s.
    dp = [0] * (n + 1) 
    # Base case: dp[0] = 0, as an empty prefix "" has 0 extra characters.

    # Iterate i from 1 to n. dp[i] corresponds to prefix s[0...i-1].
    for i in range(1, n + 1):
        # Option 1: The character s[i-1] (the i-th character of s, 0-indexed) 
        # is an extra character.
        # In this case, min extra chars for s[0...i-1] is 
        # min extra chars for s[0...i-2] (which is dp[i-1]) plus 1 (for s[i-1]).
        dp[i] = dp[i-1] + 1
        
        # Option 2: The prefix s[0...i-1] ends with a word from the dictionary.
        # This word is s[j...i-1] (which is s[j:i] in Python slicing).
        # j is the starting index of this word in s (0-indexed).
        # The prefix before this word is s[0...j-1]. The min extra characters for
        # this prefix s[0...j-1] is dp[j].
        # The word s[j:i] itself contributes 0 extra characters.
        # So, if s[j:i] is in dictionary_set, we have a candidate value dp[j] for dp[i].
        
        # We iterate j from 0 to i-1.
        # This means checking all possible substrings s[j:i] that end at index i-1.
        # e.g., if i=4, we check s[0:4], s[1:4], s[2:4], s[3:4].
        for j in range(i):  # j iterates from 0 to i-1
            # The substring/word candidate is s[j...i-1]
            word_candidate = s[j:i] 
            
            if word_candidate in dictionary_set:
                # If s[j:i] is a dictionary word, we can form s[0...i-1] by
                # taking an optimal solution for s[0...j-1] (cost dp[j])
                # and appending the word s[j:i] (cost 0 for this part).
                # We take the minimum over all such possibilities.
                dp[i] = min(dp[i], dp[j])
            
    # The final answer is dp[n], which is min extra characters for the entire string s (s[0...n-1]).
    return dp[n]