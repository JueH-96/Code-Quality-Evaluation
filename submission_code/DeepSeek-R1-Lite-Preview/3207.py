class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the minimum length among the three strings
        min_len = min(len(s1), len(s2), len(s3))
        
        # Initialize the length of the common prefix
        prefix_length = 0
        
        # Iterate through each character position up to the minimum length
        for i in range(min_len):
            # If all three characters at position i are the same, increment the prefix length
            if s1[i] == s2[i] == s3[i]:
                prefix_length += 1
            else:
                break  # Mismatch found, stop checking further
        
        # If there is no common prefix, return -1
        if prefix_length == 0:
            return -1
        
        # Calculate the total number of deletions required
        deletions = (len(s1) - prefix_length) + (len(s2) - prefix_length) + (len(s3) - prefix_length)
        
        return deletions