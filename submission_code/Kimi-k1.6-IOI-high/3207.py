class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_len = min(len(s1), len(s2), len(s3))
        for L in range(min_len, 0, -1):
            prefix1 = s1[:L]
            prefix2 = s2[:L]
            prefix3 = s3[:L]
            if prefix1 == prefix2 and prefix2 == prefix3:
                return (len(s1) - L) + (len(s2) - L) + (len(s3) - L)
        return -1