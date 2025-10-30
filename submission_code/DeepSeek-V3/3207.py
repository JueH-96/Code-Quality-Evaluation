class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the minimum length among the three strings
        min_len = min(len(s1), len(s2), len(s3))
        
        # Find the longest common prefix
        common_prefix = ""
        for i in range(min_len):
            if s1[i] == s2[i] == s3[i]:
                common_prefix += s1[i]
            else:
                break
        
        # If no common prefix, return -1
        if not common_prefix:
            return -1
        
        # Calculate the number of operations needed
        operations = (len(s1) - len(common_prefix)) + (len(s2) - len(common_prefix)) + (len(s3) - len(common_prefix))
        
        return operations