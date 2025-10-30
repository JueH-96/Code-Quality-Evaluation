class Solution:
    def minOperations(self, k: int) -> int:
        # Calculate the minimum number of operations required to make the sum of elements of the final array greater than or equal to k
        # We start with an array [1] and can either increase an element by 1 or duplicate an element
        # The optimal strategy is to increase the first element to the maximum possible value and then duplicate it
        
        # Calculate the number of operations required to increase the first element to the maximum possible value
        # This is done by subtracting 1 from k, because we start with an array [1]
        # Then we divide the result by the number of elements in the array, which is 1 in this case
        # We use the ceiling function to round up to the nearest integer, because we can't perform a fraction of an operation
        import math
        operations = math.ceil((k - 1) / 1)
        
        # If the number of operations is equal to k, it means that we need to duplicate the element
        # In this case, we subtract 1 from the number of operations, because we can duplicate the element instead of increasing it
        if operations == k:
            operations -= 1
        
        # Return the minimum number of operations required
        return operations