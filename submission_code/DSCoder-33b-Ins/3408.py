class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_chars = set(c for c in word if c.islower())
        upper_chars = set(c for c in word if c.isupper())
        return len(lower_chars & upper_chars)