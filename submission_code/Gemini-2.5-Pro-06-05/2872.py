from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        """
        Finds the largest element possible in the final array after any number of merge operations.

        The core idea is to process the array from right to left. This greedy approach
        is optimal because to maximize the value of an element by merging, we need its
        right neighbor to be as large as possible. By working backwards, we ensure that
        when we consider merging nums[i] into its right neighbor, the right neighbor's
        value has already been maximized by merges from its own right side.

        We maintain a variable `current_sum` that represents the accumulated value of the
        rightmost contiguous block being formed.
        """
        n = len(nums)

        # According to constraints, 1 <= n.
        # If n == 1, the loop doesn't run, and nums[0] is returned correctly.
        
        current_sum = nums[n - 1]
        
        # Iterate from the second-to-last element backwards to the first.
        for i in range(n - 2, -1, -1):
            # If the current number is less than or equal to the accumulated
            # sum to its right, we can merge it.
            if nums[i] <= current_sum:
                current_sum += nums[i]
            # Otherwise, the merge chain is broken. The current number must
            # start a new block.
            else:
                current_sum = nums[i]
        
        # As we iterate from right to left, the `current_sum` is always non-decreasing.
        # Thus, the final value of `current_sum` after the loop will be the maximum
        # possible value that can be formed.
        return current_sum