from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Sort the array in ascending order.
        # This is crucial because the operation specifies removing "the smallest element".
        # By sorting, we ensure that all elements that are less than 'k'
        # will appear at the beginning of the array, allowing us to count them
        # efficiently as necessary operations.
        nums.sort()
        
        operations_count = 0
        
        # Iterate through the sorted array.
        # We want to find how many initial elements are less than 'k'.
        # Each such element represents one operation (removal of the smallest element)
        # until that element is gone.
        for num in nums:
            # If the current element is less than 'k', it means it must be removed
            # to satisfy the condition that all elements are >= k.
            # This counts as one operation.
            if num < k:
                operations_count += 1
            else:
                # If the current element is greater than or equal to 'k',
                # then because the array is sorted, all subsequent elements
                # will also be greater than or equal to 'k'.
                # This means we have successfully accounted for all elements
                # that needed to be removed. We can stop counting operations,
                # as the remaining elements already satisfy the condition.
                break
                
        return operations_count