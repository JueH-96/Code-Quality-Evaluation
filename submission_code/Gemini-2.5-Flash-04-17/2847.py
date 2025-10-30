from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        """
        Calculates the maximum number of string pairs (words[i], words[j])
        where words[i] is the reverse of words[j] and i < j,
        with each string used in at most one pair.

        Args:
            words: A list of distinct strings, each of length 2.

        Returns:
            The maximum number of pairs.
        """
        # Use a set to keep track of words encountered so far that haven't been paired.
        # When we iterate through the words, if we find a word `w` whose reverse `w_rev`
        # is already in our `seen_words` set, it means `w_rev` appeared earlier in
        # the list (at an index i) and `w` appears later (at an index j > i).
        # This forms a valid pair (words[i], words[j]).
        seen_words = set()
        
        # Initialize the count of pairs
        count = 0
        
        # Iterate through each word in the input list
        for word in words:
            # Calculate the reverse of the current word
            reversed_word = word[::-1]
            
            # Check if the reversed word is in our set of seen words.
            # This means we found a word earlier in the list (at an index < current index)
            # that is the reverse of the current word.
            if reversed_word in seen_words:
                # If the reverse is found, we form a pair.
                count += 1
                # The word that was in the set (`reversed_word`) is now paired.
                # Remove it from the set so it cannot be used for another pair.
                # The current word `word` is also used in this pair and is not added to the set.
                seen_words.remove(reversed_word)
            else:
                # If the reverse is not found, the current word is not paired yet.
                # Add the current word to the set of seen words, waiting for its reverse.
                seen_words.add(word)
                
        # Return the total number of pairs found
        return count