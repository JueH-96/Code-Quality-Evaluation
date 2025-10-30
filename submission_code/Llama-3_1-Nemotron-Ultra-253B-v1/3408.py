class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_set = set()
        upper_set = set()
        for c in word:
            if c.islower():
                lower_set.add(c)
            else:
                upper_set.add(c)
        count = 0
        for char in lower_set:
            if char.upper() in upper_set:
                count += 1
        return count