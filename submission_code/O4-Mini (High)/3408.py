class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_chars = {c for c in word if c.islower()}
        upper_lower_chars = {c.lower() for c in word if c.isupper()}
        return len(lower_chars & upper_lower_chars)