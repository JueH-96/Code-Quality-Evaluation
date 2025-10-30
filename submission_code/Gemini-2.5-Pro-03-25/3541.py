import collections
from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        """
        Determines if a message contains at least two banned words.

        Args:
            message: A list of strings representing the words in the message.
            bannedWords: A list of strings representing the banned words.

        Returns:
            True if the message contains at least two words that are also in 
            bannedWords, False otherwise.
        """
        
        # Step 1: Create a set from bannedWords for efficient lookup.
        # Sets provide average O(1) time complexity for checking membership ('in' operator).
        # This is much faster than iterating through the bannedWords list for each word in message.
        # Creating the set takes O(M * K) time on average, where M is the number of banned words
        # and K is the average length of a banned word (due to hashing).
        banned_set = set(bannedWords)
        
        # Step 2: Initialize a counter for banned words found in the message.
        banned_count = 0
        
        # Step 3: Iterate through each word in the message.
        # This loop runs N times, where N is the number of words in the message.
        for word in message:
            # Step 4: Check if the current word exists in the banned_set.
            # This check takes O(L) time on average, where L is the length of the current 'word'
            # (due to hashing the word for lookup).
            if word in banned_set:
                # Step 5: If the word is banned, increment the counter.
                banned_count += 1
                
                # Step 6: Check if we have found at least two banned words.
                # If the count reaches 2, we know the message is spam according to the definition.
                # We can stop early and return True.
                if banned_count >= 2:
                    return True
                    
        # Step 7: If the loop completes without the count reaching 2,
        # it means the message is not spam. Return False.
        return False