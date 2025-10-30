class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_set = set()
        upper_set = set()

        for ch in word:
            if ch.islower():
                lower_set.add(ch)
            else:
                upper_set.add(ch.lower())

        return len(lower_set.intersection(upper_set))