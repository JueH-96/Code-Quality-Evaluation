from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # Convert the bannedWords list to a set for efficient lookups
        banned_set = set(bannedWords)
        
        # Initialize a counter to keep track of the number of banned words found
        banned_count = 0
        
        # Iterate over each word in the message
        for word in message:
            # Check if the word is in the banned set
            if word in banned_set:
                # If it is, increment the banned count
                banned_count += 1
                # If we've found at least two banned words, return True
                if banned_count >= 2:
                    return True
        
        # If we've iterated over the entire message and haven't found at least two banned words, return False
        return False