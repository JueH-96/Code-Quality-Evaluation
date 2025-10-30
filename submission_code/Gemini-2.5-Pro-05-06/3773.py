from typing import List # List is used in the type hints

class Solution:
  def minimumPairRemoval(self, nums: List[int]) -> int:
    
    # Helper function to check if the array is non-decreasing
    def is_non_decreasing(arr: List[int]) -> bool:
        # An array with 0 or 1 element is considered non-decreasing.
        # Per problem constraints, 1 <= nums.length, so arr will not be empty.
        if len(arr) <= 1:
            return True
        # Check if each element is greater than or equal to the previous one.
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                return False
        return True

    op_count = 0
    
    # The input list `nums` is modified in place.
    # Loop continues as long as `nums` is not non-decreasing.
    # The loop is guaranteed to terminate because:
    # 1. Each operation reduces the length of `nums` by 1.
    # 2. An array of length 1 is always non-decreasing.
    # Thus, `nums` will eventually become non-decreasing (at latest when len(nums) == 1).
    while not is_non_decreasing(nums):
        # Since the loop condition `not is_non_decreasing(nums)` is true,
        # `nums` must have at least 2 elements for this block to be entered.
        # If `len(nums) <= 1`, `is_non_decreasing` would have returned true, 
        # and the loop would not have started or continued.
        
        min_pair_sum = float('inf')  # Initialize with a very large number
        
        # Stores the index of the first element of the pair that has the min_pair_sum.
        # Initialized to -1, but it's guaranteed to be updated because len(nums) >= 2,
        # meaning there's at least one pair.
        idx_of_left_element_of_pair = -1 
        
        # Iterate through all adjacent pairs (nums[i], nums[i+1])
        # The loop for `i` goes from `0` to `len(nums) - 2`.
        # For example, if len(nums) is 2, range is range(1), so i will be 0.
        # This considers the pair (nums[0], nums[1]).
        for i in range(len(nums) - 1):
            current_sum = nums[i] + nums[i+1]
            
            # If this pair's sum is strictly less than the minimum found so far,
            # it becomes the new minimum. This correctly handles the "leftmost" rule:
            # if current_sum == min_pair_sum, we don't update, effectively keeping the
            # index of the earlier (leftmost) pair that achieved that sum.
            if current_sum < min_pair_sum:
                min_pair_sum = current_sum
                idx_of_left_element_of_pair = i
        
        # Perform the merging operation:
        # The element at `idx_of_left_element_of_pair` is updated to the sum.
        # Note: min_pair_sum is already nums[idx_of_left_element_of_pair] + nums[idx_of_left_element_of_pair+1]
        nums[idx_of_left_element_of_pair] = min_pair_sum 
        
        # The element at `idx_of_left_element_of_pair + 1` (the second element of the pair) is removed.
        nums.pop(idx_of_left_element_of_pair + 1)
        
        op_count += 1
            
    return op_count