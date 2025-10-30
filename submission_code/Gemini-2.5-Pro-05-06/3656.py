from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        
        # Helper function to check if elements in a list are distinct.
        def has_distinct_elements(arr_slice: List[int]) -> bool:
            # An empty list is considered to have distinct elements.
            if not arr_slice:
                return True
            # Compare the length of the list with the size of a set made from it.
            return len(set(arr_slice)) == len(arr_slice)

        # 'start_idx' tracks the beginning of the current effective array within the original 'nums'.
        start_idx = 0
        operations_count = 0

        while True:
            # The current effective array. Slicing is acceptable for N <= 100.
            current_sub_array = nums[start_idx:]
            
            # If the current array has distinct elements, we've found the minimum operations.
            if has_distinct_elements(current_sub_array):
                return operations_count
            
            # If not distinct, an operation is necessary.
            operations_count += 1
            
            # Apply the operation rules.
            if len(current_sub_array) < 3:
                # "If the array has fewer than 3 elements, remove all remaining elements."
                # This operation makes the array empty. An empty array is distinct.
                # The current 'operations_count' (which includes this final operation) is the answer.
                return operations_count 
            else:
                # "Remove 3 elements from the beginning of the array."
                # Simulate this by advancing 'start_idx'.
                start_idx += 3
            
            # The loop will continue. If 'start_idx' causes 'current_sub_array' to become empty
            # in a subsequent iteration (e.g., start_idx >= len(nums)),
            # has_distinct_elements([]) will be true, and the loop will terminate.