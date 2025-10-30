class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        special_chars_count = 0
        seen_special_chars = set()
        for char in set(word):
            if 'a' <= char <= 'z':
                if char.upper() in word and char not in seen_special_chars:
                    special_chars_count += 1
                    seen_special_chars.add(char)
            elif 'A' <= char <= 'Z':
                if char.lower() in word and char.lower() not in seen_special_chars:
                    special_chars_count += 1
                    seen_special_chars.add(char.lower())
        return special_chars_count