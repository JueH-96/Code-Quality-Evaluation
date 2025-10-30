class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        n1, n2, n3 = len(s1), len(s2), len(s3)
        
        if not (s1 and s2 and s3):
            return -1
        
        if s1[0] != s2[0] or s1[0] != s3[0]:
            return -1
        
        i = 0
        while i < min(n1, n2, n3):
            if s1[i] == s2[i] and s1[i] == s3[i]:
                i += 1
            else:
                break
        
        common_len = i
        
        return (n1 - common_len) + (n2 - common_len) + (n3 - common_len)