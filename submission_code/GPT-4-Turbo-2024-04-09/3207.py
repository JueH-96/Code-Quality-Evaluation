class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # We need to find the longest common prefix of the three strings
        # because that's the only possible way they can all be made equal
        # by removing characters from the end.
        
        # Find the minimum length among the three strings
        min_length = min(len(s1), len(s2), len(s3))
        
        # Determine the longest common prefix
        i = 0
        while i < min_length and s1[i] == s2[i] == s3[i]:
            i += 1
        
        # i is now the length of the longest common prefix
        # If i is 0 and the strings are not all empty, they can't be made equal
        if i == 0 and (len(s1) > 0 or len(s2) > 0 or len(s3) > 0):
            return -1
        
        # Calculate the total number of deletions needed
        operations = (len(s1) - i) + (len(s2) - i) + (len(s3) - i)
        return operations