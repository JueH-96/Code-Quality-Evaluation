class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        special_chars = 0
        for char in word:
            if char.islower() and char.isupper():
                special_chars += 1
        return special_chars