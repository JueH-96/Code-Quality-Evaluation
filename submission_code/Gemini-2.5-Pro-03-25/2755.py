import math
from typing import List

class Solution:
  """
  This class provides a solution to find the minimum number of extra characters 
  left over after optimally breaking a string into dictionary words.
  It uses dynamic programming to solve the problem efficiently.
  """
  def minExtraChar(self, s: str, dictionary: List[str]) -> int:
    """
    Calculates the minimum number of extra characters left over after breaking string s
    into one or more non-overlapping substrings such that each substring is present in the dictionary.
    Uses dynamic programming.

    Args:
      s: The input string (0-indexed). Length between 1 and 50.
      dictionary: A list of words. Contains distinct words. Length between 1 and 50.
                  Each word length between 1 and 50. Consists of lowercase English letters.

    Returns:
      The minimum number of extra characters left over in s if broken optimally.
    """
    
    n = len(s)
    # Convert the dictionary list to a set for efficient lookups (average O(L) time where L is word length).
    dict_set = set(dictionary)
    
    # Calculate the maximum length of a word in the dictionary. 
    # This optimization limits the search space for potential substrings.
    # Constraints state 1 <= dictionary.length, so it's guaranteed non-empty.
    max_len = 0
    if dict_set: # This check is technically redundant due to constraints but is safe practice.
         max_len = max(len(w) for w in dict_set)
    
    # Initialize the DP array `dp`. 
    # dp[i] will store the minimum number of extra characters for the prefix of s of length i (s[0...i-1]).
    # The array size is n + 1 to accommodate indices from 0 to n.
    # dp[0] = 0 because an empty prefix (length 0) has 0 extra characters.
    dp = [0] * (n + 1)
    
    # Iterate through the string `s` from index 1 to n. For each length `i`, compute dp[i].
    for i in range(1, n + 1):
        # Initialize dp[i] by considering the case where the character s[i-1] is an extra character.
        # In this scenario, the minimum extra characters count is the minimum for the prefix s[0...i-2] (which is dp[i-1]) plus 1 for s[i-1].
        dp[i] = dp[i-1] + 1
        
        # Now, consider the possibility that the prefix s[0...i-1] ends with a dictionary word.
        # Iterate through possible lengths 'k' of substrings ending at index i-1.
        # The length 'k' can range from 1 up to `max_len` (maximum dictionary word length) 
        # and cannot exceed the current prefix length `i`.
        for k in range(1, min(i, max_len) + 1):
            # The starting index of the potential substring of length k ending at i-1 is j = i - k.
            j = i - k
            # Extract the substring s[j...i-1]. In Python slicing, this is s[j:i].
            substring = s[j:i]
            
            # Check if this extracted substring is present in the dictionary set.
            if substring in dict_set:
                # If it is a dictionary word, we found a valid partition ending at i-1.
                # The cost associated with this partition is dp[j], which is the minimum extra characters
                # for the prefix s[0...j-1]. No extra characters are added for the matched word `substring`.
                # We update dp[i] if using this word partition results in fewer extra characters
                # than the current minimum stored in dp[i].
                dp[i] = min(dp[i], dp[j])
                
    # After completing the loop, dp[n] holds the minimum number of extra characters
    # for the entire string s (prefix s[0...n-1]).
    return dp[n]