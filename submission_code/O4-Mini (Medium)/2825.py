class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # Each character can never be completely removed:
        # operations only delete other occurrences of the same character,
        # leaving at least one occurrence per character.
        # Hence the minimal length is the number of distinct characters.
        return len(set(s))