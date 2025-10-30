class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        from itertools import permutations
        for p in permutations(s1):
            if p == tuple(s2):
                return True
            if p[0] == s2[0] and p[2] == s2[2]:
                if s2[1] in p and s2[3] in p:
                    return True
            if p[0] == s2[2] and p[2] == s2[0]:
                if s2[1] in p and s2[3] in p:
                    return True
        return False