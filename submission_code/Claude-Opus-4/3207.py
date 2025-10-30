class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the minimum length among the three strings
        min_len = min(len(s1), len(s2), len(s3))
        
        # Find the longest common prefix
        common_prefix_len = 0
        for i in range(min_len):
            if s1[i] == s2[i] == s3[i]:
                common_prefix_len = i + 1
            else:
                break
        
        # If no common prefix, return -1
        if common_prefix_len == 0:
            return -1
        
        # Calculate total operations needed
        # We need to remove characters from each string to make them equal to the common prefix
        operations = (len(s1) - common_prefix_len) + (len(s2) - common_prefix_len) + (len(s3) - common_prefix_len)
        
        return operations