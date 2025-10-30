class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_set = set()
        upper_set = set()
        for char in word:
            if char.islower():
                lower_set.add(char)
            elif char.isupper():
                upper_set.add(char)
        upper_lower_set = {char.lower() for char in upper_set}
        special_set = lower_set.intersection(upper_lower_set)
        return len(special_set)