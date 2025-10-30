from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # Create a set of banned words for O(1) lookups
        banned_words_set = set(bannedWords)
        
        # Initialize a counter for banned words in the message
        count = 0
        
        # Iterate through each word in the message
        for word in message:
            if word in banned_words_set:
                count += 1
                if count >= 2:
                    return True
        
        # If less than two banned words are found, return False
        return False