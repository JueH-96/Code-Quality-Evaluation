class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        n1 = len(s1)
        n2 = len(s2)
        n3 = len(s3)
        
        i = 0
        while i < min(n1, n2, n3) and s1[i] == s2[i] == s3[i]:
            i += 1
        
        if i == 0:
            return -1
        else:
            return n1 - i + n2 - i + n3 - i