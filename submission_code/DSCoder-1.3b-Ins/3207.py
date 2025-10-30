class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        def min_operations(s1, s2, s3):
            if not s1 or not s2 or not s3:
                return -1
            if len(s1) < 2 or len(s2) < 2 or len(s3) < 2:
                return -1
            if s1[-1] != s2[-1] or s1[-1] != s3[-1]:
                return -1
            return min(len(s1), len(s2), len(s3))

        return min_operations(s1, s2, s3)