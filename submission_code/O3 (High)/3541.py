from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        """
        A message array is considered spam if at least two words in it also
        appear in the bannedWords list.
        """
        # Convert bannedWords to a set for O(1) membership tests
        banned_set = set(bannedWords)
        
        match_count = 0
        for word in message:
            if word in banned_set:
                match_count += 1
                if match_count >= 2:   # Early exit as soon as condition is met
                    return True
                    
        return False