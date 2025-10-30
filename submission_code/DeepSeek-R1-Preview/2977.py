class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        if len(words) != len(s):
            return False
        acronym = ''.join(word[0] for word in words)
        return acronym == s