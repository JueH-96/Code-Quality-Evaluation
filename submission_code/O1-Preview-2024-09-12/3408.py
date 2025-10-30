class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_set = set(c for c in word if c.islower())
        upper_set = set(c.lower() for c in word if c.isupper())
        special_letters = lower_set.intersection(upper_set)
        return len(special_letters)