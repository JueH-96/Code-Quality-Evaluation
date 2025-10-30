from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ops = 0 # Count of operations performed

        # We simulate applying the operation repeatedly.
        # Each operation removes the first 3 elements from the current array.
        # After 'ops' operations, the first 'ops * 3' elements of the original array are removed.
        # The remaining part of the array starts at index 'ops * 3' in the original array.

        while True:
            # Calculate the starting index of the remaining array after 'ops' operations
            start_index = ops * 3

            # Get the subarray of the original nums starting from start_index.
            # If start_index is >= n, this slice will correctly result in an empty list [].
            current_subarray = nums[start_index:]

            # Check if the current subarray has distinct elements.
            # An empty list [] is considered to have distinct elements (len([]) == len(set([])) == 0).
            if len(current_subarray) == len(set(current_subarray)):
                # If the subarray is distinct, we've found the minimum number of operations
                # because we are incrementing 'ops' from 0 upwards.
                return ops

            # If the subarray is not distinct, perform another operation
            ops += 1