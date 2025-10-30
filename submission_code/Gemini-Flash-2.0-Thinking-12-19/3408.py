class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        special_chars_count = 0
        for char_code in range(ord('a'), ord('z') + 1):
            lower_char = chr(char_code)
            upper_char = chr(char_code - 32)
            if lower_char in word and upper_char in word:
                special_chars_count += 1
        return special_chars_count