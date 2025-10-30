from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Initialize a set with numbers from 1 to k that we need to collect.
        # A set provides efficient O(1) average time complexity for 'in' checks and 'remove' operations.
        needed_elements = set(range(1, k + 1))
        
        # Initialize a counter for the number of operations performed.
        operations_count = 0
        
        # Iterate through the array from right to left.
        # This simulates removing elements one by one from the end of the array.
        for i in range(len(nums) - 1, -1, -1):
            # Each step in the loop represents one operation.
            operations_count += 1
            
            # Get the current element from the array.
            current_num = nums[i]
            
            # If the current number is one of the elements we are looking for,
            # remove it from our set of needed elements.
            if current_num in needed_elements:
                needed_elements.remove(current_num)
            
            # Check if all needed elements have been collected (i.e., the set is empty).
            # If so, we have found the minimum number of operations.
            if not needed_elements:
                return operations_count
        
        # This part of the code should theoretically not be reached due to the problem constraint
        # that "The input is generated such that you can collect elements 1, 2, ..., k."
        # Meaning, the function is guaranteed to find all elements and return within the loop.