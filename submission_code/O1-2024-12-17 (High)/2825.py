class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # Observation:
        # Each operation removes up to two occurrences of the chosen character,
        # but never removes the chosen occurrence itself. Eventually, for each
        # character, we cannot remove its last occurrence because there would
        # be no left or right occurrence to pair with.
        #
        # Hence, at minimum, we are left with at most one occurrence of each
        # distinct character. Therefore, the minimized length is simply the
        # number of distinct characters in the string.
        
        return len(set(s))