class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_set = set()
        upper_set = set()
        for c in word:
            if c.islower():
                lower_set.add(c)
            elif c.isupper():
                upper_set.add(c.lower())
        return len(lower_set & upper_set)