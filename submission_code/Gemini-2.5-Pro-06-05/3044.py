from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum number of operations to collect elements 1 to k.

        An operation consists of removing the last element of the array. We need
        to find the minimum number of such removals until all integers from 1 to k
        have been collected.

        Args:
          nums: A list of positive integers.
          k: The target integer. We need to collect 1, 2, ..., k.

        Returns:
          The minimum number of operations required.
        """
        # A set to store the unique numbers from 1 to k that have been collected.
        # Sets provide efficient (average O(1)) additions.
        collected = set()
        
        # This will count the number of elements we remove from the end.
        operations = 0
        
        # We iterate through the list in reverse order, which is equivalent to
        # removing the last element in each step.
        for num in reversed(nums):
            # Each element we look at counts as one operation.
            operations += 1
            
            # We are only interested in numbers that are part of the target range [1, k].
            if 1 <= num <= k:
                # Add the number to our set of collected targets.
                # If the number is already in the set, this operation does nothing.
                collected.add(num)
            
            # Check if we have collected all the required k numbers.
            if len(collected) == k:
                # If we have, we've completed the task with the minimum number
                # of operations. We can return the count and stop.
                return operations
        
        # According to the problem constraints, it's always possible to collect
        # elements 1 through k. Therefore, the return statement inside the loop
        # will always be executed, and this part of the function is unreachable.
        return operations # Should not be reached