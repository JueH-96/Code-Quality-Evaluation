from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # Convert bannedWords to a set for O(1) average time complexity lookups.
        # This is crucial for performance given the constraints (up to 10^5 words).
        banned_set = set(bannedWords)
        
        spam_word_count = 0
        
        # Iterate through each word in the message.
        for word in message:
            # Check if the current word is in our set of banned words.
            if word in banned_set:
                spam_word_count += 1
                # If we've found at least two banned words, we can immediately
                # return true as the condition for spam has been met.
                if spam_word_count >= 2:
                    return True
        
        # If the loop finishes and we haven't found at least two banned words,
        # then the message is not considered spam.
        return False