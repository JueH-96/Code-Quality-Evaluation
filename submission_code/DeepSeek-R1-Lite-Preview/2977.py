from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Extract the first character of each word
        first_chars = [word[0] for word in words]
        # Concatenate the first characters
        acronym = ''.join(first_chars)
        # Compare with the given string s
        return acronym == s