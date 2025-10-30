class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        special_count = 0
        for char in set(word):
            if char.islower() and char.upper() in word:
                special_count += 1
        return special_count