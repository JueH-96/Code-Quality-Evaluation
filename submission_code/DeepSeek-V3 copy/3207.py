class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the length of the longest common prefix
        min_len = min(len(s1), len(s2), len(s3))
        common_length = 0
        for i in range(min_len):
            if s1[i] == s2[i] == s3[i]:
                common_length += 1
            else:
                break
        # If no common prefix, return -1
        if common_length == 0:
            return -1
        # Calculate the number of operations
        operations = (len(s1) - common_length) + (len(s2) - common_length) + (len(s3) - common_length)
        return operations