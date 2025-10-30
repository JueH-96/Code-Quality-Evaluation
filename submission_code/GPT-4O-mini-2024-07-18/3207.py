class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the minimum length among the three strings
        min_length = min(len(s1), len(s2), len(s3))
        
        # Initialize a variable to keep track of the common prefix length
        common_length = 0
        
        # Check for the common prefix length
        while (common_length < min_length and 
               s1[common_length] == s2[common_length] == s3[common_length]):
            common_length += 1
        
        # Calculate the number of operations needed to make all strings equal
        operations = (len(s1) - common_length) + (len(s2) - common_length) + (len(s3) - common_length)
        
        # If the common prefix length is equal to the length of any string, 
        # it means we can make them equal
        if common_length == len(s1) or common_length == len(s2) or common_length == len(s3):
            return operations
        else:
            return -1