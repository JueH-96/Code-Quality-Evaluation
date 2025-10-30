class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        if s1[0] != s2[0] or s1[0] != s3[0]:
            return -1
        
        min_length = min(len(s1), len(s2), len(s3))
        common_prefix_length = 0
        
        for i in range(min_length):
            if s1[i] == s2[i] == s3[i]:
                common_prefix_length += 1
            else:
                break
        
        return len(s1) + len(s2) + len(s3) - 3 * common_prefix_length