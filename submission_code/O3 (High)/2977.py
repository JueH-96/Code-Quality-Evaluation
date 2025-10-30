from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        """
        Build the acronym by concatenating the first character of every
        word in 'words' and compare it with 's'.
        """
        # Quick length check – if lengths differ it can’t be an acronym
        if len(words) != len(s):
            return False
        
        # Construct the acronym
        acronym = ''.join(word[0] for word in words)
        
        # Return whether constructed acronym equals s
        return acronym == s