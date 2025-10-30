from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Quick check: the number of words must match the length of s
        if len(words) != len(s):
            return False
        
        # Build the acronym from the first characters of each word
        acronym = ''.join(word[0] for word in words)
        
        # Check if it matches the input string s
        return acronym == s