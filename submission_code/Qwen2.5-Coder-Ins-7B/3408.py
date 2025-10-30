class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        special_chars = set()
        for char in word:
            if char.islower() and char.upper() in word:
                special_chars.add(char)
            elif char.isupper() and char.lower() in word:
                special_chars.add(char.lower())
        return len(special_chars)