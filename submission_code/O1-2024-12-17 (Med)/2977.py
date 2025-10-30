class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Create an acronym string by concatenating 
        # the first character of each word in words
        acronym = ''.join(word[0] for word in words)
        
        # Return True if the constructed acronym matches s, otherwise False
        return acronym == s