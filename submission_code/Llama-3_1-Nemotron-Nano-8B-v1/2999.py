class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Generate all four possible permutations of s1
        p1 = s1
        p2 = s1[2] + s1[1] + s1[0] + s1[3]
        p3 = s1[0] + s1[3] + s1[2] + s1[1]
        p4 = s1[2] + s1[3] + s1[0] + s1[1]
        return s2 in {p1, p2, p3, p4}