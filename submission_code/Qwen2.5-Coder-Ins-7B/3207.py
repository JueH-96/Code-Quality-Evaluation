class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_length = min(len(s1), len(s2), len(s3))
        i = 0
        
        while i < min_length and s1[i] == s2[i] == s3[i]:
            i += 1
        
        if i == 0:
            return -1
        
        return (len(s1) - i) + (len(s2) - i) + (len(s3) - i)