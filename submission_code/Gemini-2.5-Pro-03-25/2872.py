from typing import List

class Solution:
  """
  Represents the solution for finding the maximum array value after operations.
  """
  def maxArrayValue(self, nums: List[int]) -> int:
    """
    Calculates the largest element possible in the final array after applying the merge operation.

    The operation allows merging nums[i] into nums[i+1] if nums[i] <= nums[i+1].
    This method processes the array from right to left, greedily merging
    elements whenever possible to maximize the value of the resulting elements.

    The core idea is that to maximize the final elements, we should perform merges whenever the condition allows.
    Processing from right to left ensures that when we consider merging `nums[i]` with `nums[i+1]`, the value
    at `nums[i+1]` has potentially already been increased by merges with elements to its right. This greedy
    approach works because merging earlier (from the right) can only help satisfy the condition `nums[k] <= nums[k+1]`
    for elements further to the left, as it increases the value of `nums[k+1]`.

    Args:
        nums: A 0-indexed array of positive integers. Length >= 1.

    Returns:
        The value of the largest element that can possibly be obtained in the final array.
        The result is guaranteed to be at least max(nums).
    """
    n = len(nums)
    
    # If the array has only one element, no operations can be performed.
    # The largest element is just the element itself.
    if n == 1:
        return nums[0]

    # Initialize current_sum with the value of the last element.
    # This variable will store the sum of the current contiguous block being accumulated from the right.
    current_sum = nums[n - 1]
    
    # Initialize max_val with the value of the last element.
    # This variable will track the maximum value encountered so far, representing the largest possible element
    # in the final array configuration achieved by this greedy strategy.
    max_val = nums[n - 1]

    # Iterate through the array from the second-to-last element (index n-2) down to the first element (index 0).
    for i in range(n - 2, -1, -1):
        current_num = nums[i]
        
        # Check if the current number `nums[i]` is less than or equal to the `current_sum`.
        # `current_sum` represents the value of the element effectively at index `i+1` after potential merges
        # to its right have been performed.
        if current_num <= current_sum:
            # If the condition `nums[i] <= effective_nums[i+1]` holds, we can perform the merge operation.
            # Add the current number to the running sum.
            current_sum += current_num
        else:
            # If `nums[i] > current_sum`, we cannot merge `nums[i]` with the block to its right.
            # This means `nums[i]` starts a new potential block.
            # Reset the running sum to the current number.
            current_sum = current_num
        
        # Update the overall maximum value found so far.
        # The `current_sum` represents the value of a potential element in the final array configuration.
        # We want the largest such value possible.
        max_val = max(max_val, current_sum)

    # After iterating through all elements, max_val holds the largest possible element value achievable.
    return max_val