class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_len = min(len(s1), len(s2), len(s3))
        for L in range(min_len, 0, -1):
            if s1[:L] == s2[:L] and s1[:L] == s3[:L]:
                return (len(s1) - L) + (len(s2) - L) + (len(s3) - L)
        return -1