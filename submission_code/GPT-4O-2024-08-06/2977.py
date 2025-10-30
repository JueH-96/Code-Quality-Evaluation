class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Create the acronym by concatenating the first character of each word
        acronym = ''.join(word[0] for word in words)
        # Check if the created acronym matches the given string s
        return acronym == s