class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The minimized length is the number of unique characters in the string
        return len(set(s))