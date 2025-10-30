class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # Each character can be reduced to at most one survivor.
        # Different letters do not interact, so the minimized length
        # is just the count of distinct characters.
        return len(set(s))