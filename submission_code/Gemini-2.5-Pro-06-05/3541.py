from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # For efficient lookups, convert the list of banned words into a set.
        # This allows checking if a word is banned in O(1) average time.
        banned_set = set(bannedWords)
        
        # Initialize a counter for the number of banned words found in the message.
        count = 0
        
        # Iterate through each word in the message.
        for word in message:
            # Check if the current word exists in the set of banned words.
            if word in banned_set:
                # If it's a banned word, increment the counter.
                count += 1
            
            # A message is considered spam if it contains at least two banned words.
            # As soon as the count reaches 2, we can return True immediately
            # without processing the rest of the message. This is a key optimization.
            if count >= 2:
                return True
        
        # If the loop finishes and the count is less than 2, the message is not spam.
        return False