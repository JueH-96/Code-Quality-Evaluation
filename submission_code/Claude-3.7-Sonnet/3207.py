class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find length of common prefix
        i = 0
        min_len = min(len(s1), len(s2), len(s3))
        
        while i < min_len and s1[i] == s2[i] == s3[i]:
            i += 1
        
        # If no common prefix, return -1
        if i == 0:
            return -1
        
        # Calculate total operations:
        # Total chars to remove = sum of lengths - (3 * length of common prefix)
        return len(s1) + len(s2) + len(s3) - 3 * i