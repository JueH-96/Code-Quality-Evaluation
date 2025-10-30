class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        if len(words) != len(s):
            return False
        for word, char in zip(words, s):
            if word[0] != char:
                return False
        return True