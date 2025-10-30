class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if after possible swaps, s1 can be transformed to s2
        # Allowed swaps: swap indices 0 and 2, or 1 and 3
        # There are 4 possible combinations of swaps
        candidates = [
            s1,
            s1[2] + s1[1] + s1[0] + s1[3],  # Swap 0 and 2
            s1[0] + s1[3] + s1[2] + s1[1],  # Swap 1 and 3
            s1[2] + s1[3] + s1[0] + s1[1],  # Swap both pairs
        ]
        return s2 in candidates