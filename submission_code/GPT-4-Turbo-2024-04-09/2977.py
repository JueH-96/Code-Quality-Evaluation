class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Generate the acronym by concatenating the first character of each word
        acronym = ''.join(word[0] for word in words)
        
        # Check if the generated acronym matches the given string s
        return acronym == s