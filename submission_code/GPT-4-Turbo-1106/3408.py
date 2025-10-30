class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        return len({char.lower() for char in word if char.lower() in word and char.upper() in word})