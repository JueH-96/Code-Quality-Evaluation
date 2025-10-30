class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_set = set(word.lower())
        upper_set = set(word.upper())
        special_chars = lower_set.intersection(lower_set, upper_set)
        return len(special_chars)