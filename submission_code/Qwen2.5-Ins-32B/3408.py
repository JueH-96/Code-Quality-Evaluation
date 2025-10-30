class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        special_chars = set()
        for char in word:
            if char.lower() in word and char.upper() in word:
                special_chars.add(char.lower())
        return len(special_chars)