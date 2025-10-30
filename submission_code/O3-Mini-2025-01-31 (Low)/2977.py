class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # First check: if lengths don't match, it's impossible to form s.
        if len(words) != len(s):
            return False
        
        # Form the acronym by concatenating the first character of each word.
        acronym = ''.join(word[0] for word in words)
        
        # Compare the formed acronym with the given string s.
        return acronym == s