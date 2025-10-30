from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum number of operations to collect elements 1, 2, ..., k.
        An operation consists of removing the last element of the array and adding it
        to a collection.

        Args:
            nums: A list of positive integers.
            k: An integer representing the upper bound of the range of numbers
               to collect (1 to k).

        Returns:
            The minimum number of operations required.
        """
        # Create a set of the numbers we need to collect (1 to k).
        # Using a set allows for efficient O(1) average time complexity
        # for checking membership and removing elements.
        needed = set(range(1, k + 1))
        
        # Initialize a counter for the number of operations performed.
        operations = 0
        
        # We simulate the operations by processing the array from the end.
        # We continue performing operations (removing the last element)
        # until our collection contains all numbers from 1 to k.
        # This is represented by the 'needed' set becoming empty.
        # The problem guarantees that it is possible to collect all required numbers.
        while needed:
            # Each iteration represents one operation.
            operations += 1
            
            # Remove the last element from the 'nums' array.
            # nums.pop() removes and returns the last element of the list.
            # This directly models the operation described in the problem.
            current_num = nums.pop()
            
            # Check if the number just removed is one of the numbers we still need to collect.
            if current_num in needed:
                # If it is, we have successfully collected this number.
                # Remove it from our set of needed numbers.
                needed.remove(current_num)
                
            # The loop condition 'while needed:' ensures that we stop
            # as soon as the 'needed' set becomes empty, meaning we have
            # collected all numbers from 1 to k. This guarantees that
            # we find the minimum number of operations because we stop
            # at the earliest possible point (processing from the end).
        
        # Once the loop terminates, 'operations' holds the total count of
        # operations performed to collect all numbers from 1 to k.
        return operations