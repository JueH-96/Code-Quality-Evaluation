import collections

class Solution:
    def minimumOperations(self, nums: list[int]) -> int:
        """
        Calculates the minimum number of operations to make all elements in nums divisible by 3.
        An operation consists of adding or subtracting 1 from any element.
        """
        
        # Initialize a counter for the total number of operations.
        operations_count = 0

        # The operations on each number are independent. We can find the minimum
        # operations for each number and sum them up to get the total minimum.
        for num in nums:
            # For a single number, we want to find the minimum operations to
            # make it divisible by 3. This is equivalent to finding the
            # minimum distance to the nearest multiple of 3.

            # If num % 3 is 1 (e.g., 4), subtracting 1 (1 op) is better than adding 2 (2 ops).
            # If num % 3 is 2 (e.g., 5), adding 1 (1 op) is better than subtracting 2 (2 ops).
            # If num % 3 is 0, no operations are needed.
            
            # Thus, if a number is not perfectly divisible by 3, it takes exactly 1 operation.
            if num % 3 != 0:
                operations_count += 1
        
        return operations_count