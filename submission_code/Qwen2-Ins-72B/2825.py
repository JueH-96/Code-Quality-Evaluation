class Solution:
    def minimizedStringLength(self, s: str) -> int:
        unique_chars = set(s)
        return len(unique_chars)