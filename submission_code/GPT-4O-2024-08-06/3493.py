class Solution:
    def maxOperations(self, s: str) -> int:
        # Initialize the count of operations
        operations = 0
        # Initialize the count of zeros encountered
        zero_count = 0
        
        # Traverse the string
        for char in s:
            if char == '0':
                # Count the zeros
                zero_count += 1
            elif char == '1':
                # If we encounter a '1' and there are zeros before it,
                # we can perform an operation
                if zero_count > 0:
                    operations += 1
                    # After performing the operation, we reduce the zero count
                    zero_count -= 1
        
        return operations