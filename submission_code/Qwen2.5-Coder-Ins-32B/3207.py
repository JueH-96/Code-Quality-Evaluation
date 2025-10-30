class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_length = min(len(s1), len(s2), len(s3))
        common_length = 0
        
        for i in range(min_length):
            if s1[i] == s2[i] == s3[i]:
                common_length += 1
            else:
                break
        
        if common_length == 0:
            return -1
        
        total_length = len(s1) + len(s2) + len(s3)
        return total_length - 3 * common_length