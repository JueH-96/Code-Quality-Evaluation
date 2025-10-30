from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        """
        Calculates the maximum number of pairs of strings that can be formed from a given list,
        where a pair can be formed by two strings if one is the reverse of the other.

        Args:
            words: A list of distinct strings.

        Returns:
            The maximum number of pairs that can be formed.
        """
        
        # A set to store words encountered so far for efficient O(1) average time lookups.
        seen = set()
        
        # A counter for the number of pairs found.
        count = 0
        
        # Iterate through each word in the list.
        for word in words:
            # The reversed version of the current word is its potential partner.
            reversed_word = word[::-1]
            
            # If the reversed word has been seen before, we have found a pair.
            if reversed_word in seen:
                count += 1
            else:
                # If its partner has not been seen, add the current word to the set.
                # This makes it available for a future word to be its partner.
                seen.add(word)
                
        return count