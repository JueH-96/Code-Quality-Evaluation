from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        
        # If the array has only one element, that's the largest possible value.
        if n == 1:
            return nums[0]
        
        # Initialize current_sum with the rightmost element. This variable will store the
        # sum of the current potential block of elements that can be merged together,
        # starting from the right and moving left.
        # It represents the value at the position of the rightmost element in the current potential merge block
        # after potential merges from its original index onwards.
        # max_val will track the maximum value encountered among all such blocks formed.
        
        # Python's int handles arbitrary size, so no need for explicit long.
        current_sum: int = nums[n - 1]
        max_val: int = nums[n - 1] # The rightmost element is initially the max candidate

        # Iterate from the second to last element (index n-2) down to the first element (index 0).
        for i in range(n - 2, -1, -1):
            # Check if the element at the current index i can be merged into the block
            # represented by current_sum (which is conceptually to its right, at index i+1
            # in the original array's layout, potentially containing a sum of elements
            # from i+1 onwards).
            if nums[i] <= current_sum:
                # If nums[i] is less than or equal to the current_sum, we can merge it.
                # Add nums[i] to the current_sum. The combined value now represents
                # a larger potential merged block starting further left (at original index i).
                current_sum += nums[i]
            else:
                # If nums[i] is greater than the current_sum, nums[i] cannot be merged
                # into the block to its right. The block represented by the current_sum
                # is effectively "terminated" at its left boundary by nums[i].
                # nums[i] now becomes the start of a *new* potential merged block.
                current_sum = nums[i]
            
            # After updating current_sum (either by adding or resetting), the new
            # current_sum value is a candidate for the largest element possible.
            # Update the overall maximum value found so far.
            max_val = max(max_val, current_sum)

        # After iterating through all elements, max_val holds the largest sum found.
        return max_val