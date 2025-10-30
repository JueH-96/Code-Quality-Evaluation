from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # The acronym must have the same length as the number of words
        if len(s) != len(words):
            return False
        
        # Build the acronym from the first letters of each word
        acronym = ''.join(word[0] for word in words)
        
        # Check if it matches s
        return acronym == s