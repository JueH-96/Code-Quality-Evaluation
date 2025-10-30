class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_len = min(len(s1), len(s2), len(s3))
        for t in range(min_len, 0, -1):
            if s1[:t] == s2[:t] == s3[:t]:
                return (len(s1) - t) + (len(s2) - t) + (len(s3) - t)
        return -1