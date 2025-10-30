from typing import List # Add this import line before the class definition

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        """
        Calculates the maximum number of pairs of strings that can be formed from the array 
        such that one string is the reverse of the other.

        The problem specifies that words[i] can be paired with words[j] if words[i] == reversed(words[j]) 
        and 0 <= i < j < words.length. Each string can belong to at most one pair.

        The approach uses a set to efficiently track words encountered so far that have not yet been paired.
        For each word processed from the input list, it checks if its reverse exists in the set of 
        previously seen (and unpaired) words. 

        If the reverse is found in the set, it means a pair can be formed between the current word 
        and the previously seen reversed word. This satisfies the i < j condition because the reversed word 
        must have appeared earlier in the list to be in the set. We increment the pair count.
        Since each word can only be in one pair, we conceptually "use up" both words. We achieve this by
        incrementing the count and *not* adding the current word to the set. The previously seen reversed word 
        remains in the set, but due to the distinctness constraint of words in the input, it cannot form 
        another pair.

        If the reverse is not found in the set, the current word cannot form a pair with any previously seen word. 
        We add the current word to the set, making it available to potentially form a pair with a word 
        encountered later in the list that is its reverse.

        Args:
            words: A list of distinct strings, where each string has a length of 2 
                   and contains only lowercase English letters. Constraints guarantee 
                   1 <= words.length <= 50.

        Returns:
            The maximum number of pairs that can be formed according to the specified criteria.
            This corresponds to the number of times we find a word whose reverse has been seen previously.
        """
        
        # Initialize the count of pairs found
        pair_count = 0
        
        # Initialize a set to store words that have been encountered but not yet paired.
        # Using a set provides O(1) average time complexity for checking membership ('in' operator)
        # and adding elements ('add' method).
        seen_words = set()

        # Iterate through each word in the input list 'words'.
        for word in words:
            # Calculate the reversed version of the current word.
            # String slicing `[::-1]` is an efficient way to reverse a string in Python.
            # For example, "ab"[::-1] yields "ba", "zz"[::-1] yields "zz".
            reversed_word = word[::-1]

            # Check if the reversed version of the current word is already present in our set 'seen_words'.
            # This means we have encountered the complement of the current word earlier in the list.
            if reversed_word in seen_words:
                # If the reversed word is found in the set, it signifies that we have found a valid pair.
                # The pair consists of 'reversed_word' (which appeared at index i) and 'word' (at index j > i).
                # Increment the count of pairs found.
                pair_count += 1
                
                # We don't add the current 'word' to the 'seen_words' set because it has now been used in a pair.
                # We also do not need to remove 'reversed_word' from 'seen_words'. 
                # Although it is also part of the pair, leaving it in the set causes no harm. 
                # If another word 'w' were the reverse of 'reversed_word', then 'w' would equal 'word'.
                # But the input 'words' contains distinct strings, so this cannot happen.
            else:
                # If the reversed word is not found in the set 'seen_words', it means the current 'word' 
                # has not yet found its matching pair among the words processed so far.
                # Add the current 'word' to the 'seen_words' set. This makes it available 
                # to be potentially matched by a future word in the list that is its reverse.
                seen_words.add(word)

        # After iterating through all the words in the list, return the total count of pairs found.
        return pair_count