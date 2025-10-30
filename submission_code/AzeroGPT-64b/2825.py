class Solution:
    def minimizedStringLength(self, s: str) -> int:
        uniqueChars = set(s)  # Get unique characters in the string
        return len(uniqueChars)  # Return the count of unique characters, as any character can appear at most once in the minimized string.