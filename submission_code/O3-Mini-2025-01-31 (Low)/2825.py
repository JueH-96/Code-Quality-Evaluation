class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The optimal minimized string simply retains one occurrence per unique character.
        # Since every operation removes one occurrence of a repeated character,
        # we can eventually remove all duplicates.
        # Hence, the minimized string length is the number of distinct characters in s.
        return len(set(s))