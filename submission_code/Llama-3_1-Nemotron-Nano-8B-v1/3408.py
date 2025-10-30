class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_chars = set()
        upper_chars = set()
        for c in word:
            if c.islower():
                lower_chars.add(c)
            else:
                upper_chars.add(c)
        count = 0
        for c in lower_chars:
            if c.upper() in upper_chars:
                count += 1
        return count