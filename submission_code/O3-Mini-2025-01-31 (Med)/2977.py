class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # If lengths do not match, s cannot be the acronym.
        if len(words) != len(s):
            return False
        
        # Build the acronym by taking the first letter of each word.
        acronym = ''.join(word[0] for word in words)
        
        # Check if the built acronym is equal to s.
        return acronym == s