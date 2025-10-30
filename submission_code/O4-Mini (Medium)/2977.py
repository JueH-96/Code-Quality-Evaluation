from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # The acronym must have the same length as the number of words
        if len(s) != len(words):
            return False
        
        # Check that each character in s matches the first character of the corresponding word
        for i, word in enumerate(words):
            if s[i] != word[0]:
                return False
        
        return True