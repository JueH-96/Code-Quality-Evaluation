class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # We can use a set to track unique characters in the string.
        # The minimized length of the string is the number of unique characters.
        unique_chars = set(s)
        return len(unique_chars)