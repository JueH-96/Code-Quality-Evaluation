class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        special_chars = set()
        for char in word:
            if char.isupper() and char.lower() in word:
                special_chars.add(char)
        return len(special_chars)