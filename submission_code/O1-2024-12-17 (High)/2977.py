class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Create a string by concatenating the first character of each word
        acronym = "".join(word[0] for word in words)
        # Compare with s
        return acronym == s