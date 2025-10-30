class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Determine the minimum length among the three strings
        min_len = min(len(s1), len(s2), len(s3))
        l = 0
        
        # Find the longest common prefix length
        for i in range(min_len):
            if s1[i] == s2[i] == s3[i]:
                l += 1
            else:
                break
        
        # If no common prefix exists at all
        if l == 0:
            return -1
        
        # Calculate the total operations needed
        return (len(s1) + len(s2) + len(s3)) - 3 * l