class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # All possible configurations reachable from s1
        possible = {
            s1,                              # original
            s1[2] + s1[1] + s1[0] + s1[3],  # swap indices 0 and 2
            s1[0] + s1[3] + s1[2] + s1[1],  # swap indices 1 and 3
            s1[2] + s1[3] + s1[0] + s1[1]   # both swaps
        }
        return s2 in possible