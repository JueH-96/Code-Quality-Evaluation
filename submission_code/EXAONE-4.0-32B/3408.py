class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        char_set = set(word)
        count = 0
        for c in "abcdefghijklmnopqrstuvwxyz":
            if c in char_set and c.upper() in char_set:
                count += 1
        return count