class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        if not (s1[0] == s2[0] == s3[0]):
            return -1

        n1 = len(s1)
        n2 = len(s2)
        n3 = len(s3)

        min_len = min(n1, n2, n3)

        for l in range(min_len, 0, -1):
            prefix = s1[:l]
            if s2[:l] == prefix and s3[:l] == prefix:
                operations = (n1 - l) + (n2 - l) + (n3 - l)
                return operations

        return -1