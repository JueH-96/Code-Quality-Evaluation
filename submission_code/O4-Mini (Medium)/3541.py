from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # Convert bannedWords to a set for O(1) lookups
        banned_set = set(bannedWords)
        
        # Count how many words in message appear in banned_set
        count = 0
        for word in message:
            if word in banned_set:
                count += 1
                # As soon as we hit two, it's spam
                if count >= 2:
                    return True
        
        # If fewer than two matches, not spam
        return False