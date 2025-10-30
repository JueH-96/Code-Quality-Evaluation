class Solution:
    def minimizedStringLength(self, s: str) -> int:
        n = len(s)
        chars = set()
        for char in s:
            chars.add(char)
        return len(chars)