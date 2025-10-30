class Solution:
    def isAcronym(self, words: list[str], s: str) -> bool:
        acronym = ''.join(word[0] for word in words)
        return acronym == s