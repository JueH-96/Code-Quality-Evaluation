class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        n1 = len(s1)
        n2 = len(s2)
        n3 = len(s3)
        common_length = 0
        for i in range(min(n1, n2, n3)):
            if s1[i] == s2[i] and s2[i] == s3[i]:
                common_length += 1
            else:
                break
        if common_length == 0:
            if s1[0] != s2[0] or s1[0] != s3[0] or s2[0] != s3[0]:
                return -1
            else:
                return 0 if (n1 == n2 == n3 == 1 and s1[0] == s2[0] == s3[0]) else -1


        ops1 = n1 - common_length
        ops2 = n2 - common_length
        ops3 = n3 - common_length
        return ops1 + ops2 + ops3