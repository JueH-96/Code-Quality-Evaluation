from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        operations_count = 0
        
        # This set will store the numbers from 1 to k that we still need to find.
        # Initially, it contains all numbers from 1 to k.
        # Example: if k=3, needed_elements will be {1, 2, 3}.
        needed_elements = set(range(1, k + 1))
        
        # We simulate the operations by "removing" elements from the end of the nums array.
        # The loop continues as long as 'needed_elements' is not empty,
        # meaning there are still target numbers (1 to k) that we haven't collected.
        # The problem guarantees that it's always possible to collect elements 1 through k,
        # so nums will not become empty before needed_elements is empty.
        while needed_elements:  # This condition is true as long as the set is not empty
            operations_count += 1
            
            # Perform the operation: remove the last element from nums.
            # list.pop() removes and returns the last item in O(1) time.
            current_element = nums.pop()
            
            # Check if the removed element is one of the numbers we are looking for (1 to k)
            # AND if we haven't already "collected" it (i.e., it's still in needed_elements).
            # The 'in' check for a set is O(1) on average.
            if current_element in needed_elements:
                # If it is, we've now "collected" it, so remove it from the set of needed elements.
                # set.remove() is O(1) on average.
                needed_elements.remove(current_element)
            
        # Once 'needed_elements' is empty, we have collected all numbers from 1 to k.
        # The value of 'operations_count' at this point is the minimum number of operations.
        return operations_count