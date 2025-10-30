class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the length of the longest common prefix
        common_prefix_length = 0
        min_length = min(len(s1), len(s2), len(s3))
        
        for i in range(min_length):
            if s1[i] == s2[i] == s3[i]:
                common_prefix_length += 1
            else:
                break
        
        # If no common prefix, it's impossible
        if common_prefix_length == 0:
            return -1
        
        # Calculate total deletions needed
        return (len(s1) + len(s2) + len(s3)) - 3 * common_prefix_length