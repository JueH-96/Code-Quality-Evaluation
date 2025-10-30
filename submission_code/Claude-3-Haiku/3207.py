class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        if len(s1) + len(s2) + len(s3) % 3 != 0:
            return -1
        
        target_length = (len(s1) + len(s2) + len(s3)) // 3
        
        if len(s1) < target_length or len(s2) < target_length or len(s3) < target_length:
            return -1
        
        operations = 0
        
        while len(s1) > target_length or len(s2) > target_length or len(s3) > target_length:
            if len(s1) >= len(s2) and len(s1) >= len(s3):
                s1 = s1[:-1]
            elif len(s2) >= len(s1) and len(s2) >= len(s3):
                s2 = s2[:-1]
            else:
                s3 = s3[:-1]
            operations += 1
        
        if s1 == s2 == s3:
            return operations
        else:
            return -1