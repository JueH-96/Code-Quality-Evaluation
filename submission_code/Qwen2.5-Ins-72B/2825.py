class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The minimized string length is equal to the number of unique characters in the string
        return len(set(s))