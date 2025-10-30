class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Helper function to find the minimum number of operations to make two strings equal
        def min_operations_to_equal(s1, s2):
            while s1 and s2 and s1[-1] == s2[-1]:
                s1 = s1[:-1]
                s2 = s2[:-1]
            return len(s1) + len(s2)
        
        # Check if there is a way to make the strings equal
        if s1[0] != s2[0] or s2[0] != s3[0] or s1[0] != s3[0]:
            return -1
        
        # Calculate the minimum number of operations for each pair of strings
        ops1 = min_operations_to_equal(s1, s2)
        ops2 = min_operations_to_equal(s2, s3)
        ops3 = min_operations_to_equal(s1, s3)
        
        # Return the sum of minimum operations for each pair
        return (ops1 + ops2 + ops3) // 2