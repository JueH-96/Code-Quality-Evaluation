from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        pair_count = 0
        seen_words = set() # Use a set to store words encountered so far, waiting for a pair

        for word in words:
            # Calculate the reversed version of the current word
            reversed_word = word[::-1]

            # Check if the reversed word has been seen before and is available for pairing
            if reversed_word in seen_words:
                # If found, we've formed a pair!
                pair_count += 1
                
                # Remove the 'reversed_word' from the set because it has now been used
                # in a pair. The current 'word' is also considered used and is not added
                # back to the set.
                seen_words.remove(reversed_word)
            else:
                # If the reversed word is not found, add the current word to the set
                # It's waiting for its potential match later in the list.
                seen_words.add(word)
        
        return pair_count