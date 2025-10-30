class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # Use a set to keep track of unique characters in the string
        unique_chars = set(s)
        # The length of the minimized string is the number of unique characters
        return len(unique_chars)