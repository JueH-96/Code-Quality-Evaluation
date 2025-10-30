class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the length of the common prefix
        i = 0
        min_len = min(len(s1), len(s2), len(s3))
        while i < min_len and s1[i] == s2[i] == s3[i]:
            i += 1
        
        # If there's no common prefix, return -1
        if i == 0:
            return -1
        
        # Calculate the total number of characters to remove
        total_to_remove = (len(s1) - i) + (len(s2) - i) + (len(s3) - i)
        
        return total_to_remove