class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        n1, n2, n3 = len(s1), len(s2), len(s3)
        min_len = min(n1, n2, n3)
        L0 = 0
        while L0 < min_len and s1[L0] == s2[L0] == s3[L0]:
            L0 += 1
        if L0 == 0:
            return -1
        return (n1 - L0) + (n2 - L0) + (n3 - L0)