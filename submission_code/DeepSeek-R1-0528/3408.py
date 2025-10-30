class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        count = 0
        for c in "abcdefghijklmnopqrstuvwxyz":
            if c in word and c.upper() in word:
                count += 1
        return count