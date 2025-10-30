class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_len = min(len(s1), len(s2), len(s3))
        
        for l in range(min_len, 0, -1):
            if s1[:l] == s2[:l] == s3[:l]:
                operations = (len(s1) - l) + (len(s2) - l) + (len(s3) - l)
                return operations
        
        return -1