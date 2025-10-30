class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        if sorted(s1) != sorted(s2):
            return False
        if s1[0] != s2[0] and s1[0] != s2[2]:
            return False
        if s1[1] != s2[1] and s1[1] != s2[3]:
            return False
        if s1[2] != s2[2] and s1[2] != s2[0]:
            return False
        if s1[3] != s2[3] and s1[3] != s2[1]:
            return False

        return True