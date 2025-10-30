class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # If first characters are not equal, impossible to make strings equal
        if s1[0] != s2[0] or s2[0] != s3[0]:
            return -1
            
        # Find length of common prefix
        min_len = min(len(s1), len(s2), len(s3))
        common_len = 0
        
        for i in range(min_len):
            if s1[i] == s2[i] == s3[i]:
                common_len += 1
            else:
                break
                
        # If no common prefix except first char, impossible
        if common_len == 0:
            return -1
            
        # Operations needed = total extra chars to remove from all strings
        total_ops = len(s1) + len(s2) + len(s3) - (3 * common_len)
        
        return total_ops