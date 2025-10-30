class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find longest common prefix length among s1, s2, s3
        min_len = min(len(s1), len(s2), len(s3))
        k = 0
        for i in range(min_len):
            if s1[i] == s2[i] == s3[i]:
                k += 1
            else:
                break
        
        # If the common prefix is empty, we cannot delete down to empty (deleting last char from length 1 is disallowed)
        if k == 0:
            return -1
        
        # Otherwise each string must be reduced to length k
        return (len(s1) - k) + (len(s2) - k) + (len(s3) - k)