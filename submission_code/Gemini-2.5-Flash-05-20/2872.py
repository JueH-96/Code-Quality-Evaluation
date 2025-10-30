from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        
        # According to constraints, 1 <= nums.length, so nums is never empty.
        # Initialize the current_sum_candidate with the last element.
        # This element is the rightmost end of any potential merged block.
        current_sum_candidate = nums[n - 1]
        
        # Initialize max_val with the last element.
        # This will hold the maximum value found among all possible final elements.
        max_val = nums[n - 1]
        
        # Iterate from the second-to-last element down to the first element.
        # We are essentially trying to merge elements from left into the
        # current_sum_candidate (which represents a sum of elements to their right).
        for i in range(n - 2, -1, -1):
            # If the current element nums[i] is less than or equal to the current_sum_candidate,
            # it means we can merge nums[i] into the current_sum_candidate.
            # We always choose to do this because it increases the sum, which is our goal.
            # A larger sum is also more likely to be able to absorb elements further to its left.
            if nums[i] <= current_sum_candidate:
                current_sum_candidate += nums[i]
            else:
                # If nums[i] is greater than current_sum_candidate,
                # we cannot merge nums[i] into current_sum_candidate.
                # This implies that current_sum_candidate (the block ending at i+1)
                # cannot be extended by nums[i].
                # So, nums[i] must start a new potential sum block on its own.
                current_sum_candidate = nums[i]
            
            # After each decision (either merge or new block), update the overall
            # maximum value encountered so far. This accounts for scenarios where
            # a previous current_sum_candidate might be the largest, or a new
            # single element (nums[i] when starting a new block) might be the largest.
            max_val = max(max_val, current_sum_candidate)
            
        return max_val