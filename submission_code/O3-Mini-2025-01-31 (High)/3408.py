class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_chars = set()
        upper_chars = set()
        for char in word:
            if char.islower():
                lower_chars.add(char)
            elif char.isupper():
                upper_chars.add(char.lower())
        return len(lower_chars.intersection(upper_chars))