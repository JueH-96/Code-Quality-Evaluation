class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lowercase_letters = set()
        uppercase_letters = set()
        for char in word:
            if 'a' <= char <= 'z':
                lowercase_letters.add(char)
            elif 'A' <= char <= 'Z':
                uppercase_letters.add(char)
        
        special_char_count = 0
        for lower_char in lowercase_letters:
            upper_char = lower_char.upper()
            if upper_char in uppercase_letters:
                special_char_count += 1
        return special_char_count