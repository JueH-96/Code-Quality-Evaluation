from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Generate the acronym from the words list
        acronym = ''.join(word[0] for word in words)
        # Check if the generated acronym matches the string s
        return acronym == s