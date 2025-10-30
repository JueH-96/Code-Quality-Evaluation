class Solution:
    def minimizedStringLength(self, s: str) -> int:
        unique_chars = set()
        for char in s:
            unique_chars.add(char)
        return len(unique_chars)