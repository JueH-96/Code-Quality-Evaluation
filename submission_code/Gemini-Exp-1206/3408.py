class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower = set()
        upper = set()
        for char in word:
            if 'a' <= char <= 'z':
                lower.add(char)
            else:
                upper.add(char)
        count = 0
        for char in lower:
            if char.upper() in upper:
                count += 1
        return count