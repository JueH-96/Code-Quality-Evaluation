class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Generate all possible transformed versions of s1 using allowed swaps
        candidates = [
            s1,
            s1[2] + s1[1] + s1[0] + s1[3],   # swap indices 0 and 2
            s1[0] + s1[3] + s1[2] + s1[1],   # swap indices 1 and 3
            s1[2] + s1[3] + s1[0] + s1[1]    # swap both pairs (0-2 and 1-3)
        ]
        return s2 in candidates