from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # Create a set of banned words for O(1) average lookup
        # This is more efficient than checking against a list repeatedly
        banned_set = set(bannedWords)
        
        # Counter for the number of banned words found in the message
        banned_count = 0
        
        # Iterate through the message words
        for word in message:
            # Check if the current word is in the banned set
            if word in banned_set:
                banned_count += 1
                # If we find at least two banned words, the message is spam
                # We can return True immediately to save further processing
                if banned_count >= 2:
                    return True
                    
        # If the loop finishes and we haven't found two banned words, it's not spam
        return False