class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The minimized length of the string will be the number of unique characters in the string
        return len(set(s))