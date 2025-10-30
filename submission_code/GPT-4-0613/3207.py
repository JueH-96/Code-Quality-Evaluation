class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Reverse the strings to make it easier to compare from the end
        s1 = s1[::-1]
        s2 = s2[::-1]
        s3 = s3[::-1]
        
        # Initialize the count of operations
        operations = 0
        
        # While the strings are not all equal
        while s1 != s2 or s2 != s3:
            # If any of the strings is empty, return -1
            if not s1 or not s2 or not s3:
                return -1
            
            # If the last characters of the strings are not all equal
            if s1[0] != s2[0] or s2[0] != s3[0]:
                # Remove the last character from the longest string
                if len(s1) >= len(s2) and len(s1) >= len(s3):
                    s1 = s1[1:]
                elif len(s2) >= len(s1) and len(s2) >= len(s3):
                    s2 = s2[1:]
                else:
                    s3 = s3[1:]
                
                # Increment the count of operations
                operations += 1
            else:
                # If the last characters are equal, remove them from all strings
                s1 = s1[1:]
                s2 = s2[1:]
                s3 = s3[1:]
        
        # Return the count of operations
        return operations