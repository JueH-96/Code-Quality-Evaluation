class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # Use a set to keep track of unique characters
        unique_chars = set(s)
        # The minimized length is the number of unique characters
        return len(unique_chars)