from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        
        # Convert dictionary to a set for O(1) average time complexity lookups.
        # This is crucial for performance given the constraints on s.length and dictionary word lengths.
        dictionary_set = set(dictionary)
        
        # dp[i] will store the minimum number of extra characters left over
        # when considering the prefix s[0...i-1].
        # The size of dp array is n+1 because dp[0] is for an empty prefix
        # and dp[n] is for the entire string s.
        # Initialize dp array with a value greater than any possible result.
        # The maximum possible extra characters is n (if no words are found and all chars are extra).
        # So, n + 1 serves as a good initial "infinity" value.
        dp = [n + 1] * (n + 1)
        
        # Base case:
        # An empty prefix (s[0...-1]) has 0 characters, hence 0 extra characters.
        dp[0] = 0
        
        # Iterate through each possible ending index 'i' for a prefix s[0...i-1].
        # 'i' ranges from 1 to n (inclusive).
        for i in range(1, n + 1):
            # Option 1: The character s[i-1] (the last character of the current prefix)
            # is considered an "extra" character.
            # In this case, the minimum extra characters for s[0...i-1] would be
            # the minimum extra characters for s[0...i-2] (which is dp[i-1]) plus 1.
            dp[i] = dp[i-1] + 1
            
            # Option 2: The character s[i-1] is the end of a dictionary word.
            # We iterate through all possible starting positions 'j' for a word
            # that ends at s[i-1]. A word would be s[j...i-1].
            # 'j' ranges from 0 to i-1 (inclusive).
            for j in range(i):
                # Extract the substring from index `j` up to (but not including) `i`.
                # This corresponds to s[j...i-1].
                current_word = s[j:i]
                
                # Check if this substring is present in our dictionary set.
                if current_word in dictionary_set:
                    # If it is a valid dictionary word, we can use it.
                    # The number of extra characters up to this point would then be
                    # the minimum extra characters for the prefix s[0...j-1] (which is dp[j]),
                    # because the substring s[j...i-1] is fully accounted for by a dictionary word.
                    # We update dp[i] to be the minimum of its current value (from Option 1)
                    # and dp[j] (from Option 2).
                    dp[i] = min(dp[i], dp[j])
                    
        # After iterating through all prefixes, dp[n] will hold the minimum
        # number of extra characters for the entire string s.
        return dp[n]