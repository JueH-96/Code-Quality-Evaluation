class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Create the acronym by taking the first character of each word
        acronym = ''.join(word[0] for word in words)
        
        # Check if the formed acronym matches the given string s
        return acronym == s