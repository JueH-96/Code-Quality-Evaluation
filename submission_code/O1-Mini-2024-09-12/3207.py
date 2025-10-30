class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_length = min(len(s1), len(s2), len(s3))
        for l in range(min_length, 0, -1):
            prefix = s1[:l]
            if s2.startswith(prefix) and s3.startswith(prefix):
                return (len(s1) - l) + (len(s2) - l) + (len(s3) - l)
        return -1