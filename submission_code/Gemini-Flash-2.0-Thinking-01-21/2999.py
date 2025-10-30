class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        s1_02 = sorted([s1[0], s1[2]])
        s1_13 = sorted([s1[1], s1[3]])
        s2_02 = sorted([s2[0], s2[2]])
        s2_13 = sorted([s2[1], s2[3]])
        return s1_02 == s2_02 and s1_13 == s2_13