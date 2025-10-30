class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Build the acronym by taking the first character of each word
        acronym = ''.join(word[0] for word in words)
        
        # Check if the acronym matches s
        return acronym == s