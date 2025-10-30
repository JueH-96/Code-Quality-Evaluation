class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        s1_pair1 = "".join(sorted([s1[0], s1[2]]))
        s1_pair2 = "".join(sorted([s1[1], s1[3]]))
        s2_pair1 = "".join(sorted([s2[0], s2[2]]))
        s2_pair2 = "".join(sorted([s2[1], s2[3]]))
        return s1_pair1 == s2_pair1 and s1_pair2 == s2_pair2