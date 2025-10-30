class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the minimum length among the three strings
        min_length = min(len(s1), len(s2), len(s3))
        
        # Initialize the number of operations
        operations = 0
        
        # Start from the end of the strings and move towards the start
        for i in range(min_length):
            # Compare the characters from the end
            if s1[-1-i] == s2[-1-i] == s3[-1-i]:
                continue
            else:
                # If they are not equal, we need to perform operations
                # Calculate the number of operations needed to make them equal
                # We need to make two of them equal to the third one
                if s1[-1-i] != s2[-1-i] and s1[-1-i] != s3[-1-i] and s2[-1-i] != s3[-1-i]:
                    return -1
                operations += 1
        
        # Add the remaining length of the strings that need to be removed
        operations += (len(s1) - min_length) + (len(s2) - min_length) + (len(s3) - min_length)
        
        return operations