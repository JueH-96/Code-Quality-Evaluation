class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        return all(word[0] == s[i] for i, word in enumerate(words)) and len(s) == len(words[0])