class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        if len(words) != len(s):
            return False
        return all(word[0] == letter for word, letter in zip(words, s))