from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Initialize an empty string to store the acronym
        acronym = ""
        
        # Iterate over each word in the list of words
        for word in words:
            # Append the first character of the word to the acronym
            acronym += word[0]
        
        # Check if the acronym is equal to the given string
        return acronym == s