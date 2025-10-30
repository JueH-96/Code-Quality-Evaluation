class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Build an acronym by taking the first character of each word
        acronym = ''.join(word[0] for word in words)
        # Compare the constructed acronym with the given string s
        return acronym == s