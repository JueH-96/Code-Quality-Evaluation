from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Build the acronym by concatenating the first character of each word.
        acronym = ''.join(word[0] for word in words)
        return acronym == s