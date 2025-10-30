import collections
from typing import List

class Solution:
  """
  Solves the checkArray problem using a greedy approach with O(N) time and O(1) extra space.
  The problem asks if an array can be made all zeros by repeatedly
  decreasing elements of subarrays of size k by 1.
  """
  def checkArray(self, nums: List[int], k: int) -> bool:
    """
    Determines if the array can be made all zeros by repeatedly applying
    the operation: choose subarray of size k and decrease all its elements by 1.

    This implementation uses a greedy approach, processing the array from left to right.
    It maintains the cumulative decrease effect from operations started at previous indices.
    It uses O(1) extra space by modifying the input array `nums` in place to store
    the number of operations started at each index.

    Args:
      nums: The input integer array. This array is modified in place.
      k: The size of the subarray for the operation.

    Returns:
      True if the array can be made all zeros, False otherwise.
    """
    n = len(nums)
    # current_decrease represents the total decrease applied to the current element nums[i]
    # due to operations started at indices before i. Specifically, it's the sum of
    # operations `op[j]` where `max(0, i-k+1) <= j < i`.
    current_decrease = 0

    for i in range(n):
        # The effective value of nums[i] is its original value minus the cumulative decrease
        # from operations started at previous indices that affect index i.
        effective_value = nums[i] - current_decrease

        # If the effective value is negative, it means we needed to decrease nums[i] more
        # than its current value allows, which is impossible since operations only decrease values.
        if effective_value < 0:
            return False

        # The number of operations starting at index `i`, let's call it `op_i`, must be exactly
        # `effective_value`. This is because this is the last chance to decrease `nums[i]`
        # (operations starting later won't affect `nums[i]`).
        op_i = effective_value
        
        # Check the boundary condition: If an operation is needed (op_i > 0),
        # ensure it doesn't extend beyond the array bounds. The operation affects
        # the subarray `nums[i:i+k]`. If `i + k > n`, the subarray goes out of bounds.
        # This check is crucial for the last k-1 elements of the array.
        if op_i > 0 and i + k > n:
            return False

        # Update the cumulative decrease for the *next* element (i+1).
        # The operation `op_i` starts affecting elements from index `i` onwards.
        current_decrease += op_i
        
        # Store `op_i` in `nums[i]`. This allows us to retrieve this value later when the
        # effect of this operation expires (at index i+k). This in-place modification
        # allows for O(1) extra space complexity. We overwrite the original nums[i]
        # after we have used it to calculate op_i.
        # Note: the stored value represents the *operation amount*, not the resulting element value.
        nums[i] = op_i

        # An operation started k steps ago, at index `i - k + 1`, affects elements up to
        # index `(i - k + 1) + k - 1 = i`. It does *not* affect index `i+1`.
        # Therefore, we must subtract its contribution (`op[i-k+1]`) from `current_decrease`
        # when moving to the next step. This update applies only if `i - k + 1 >= 0`, which means `i >= k - 1`.
        # We retrieve `op[i-k+1]` from the value previously stored in `nums[i-k+1]`.
        if i >= k - 1:
             # Subtract the effect of the operation that just expired.
             current_decrease -= nums[i - k + 1] 

    # If the loop completes without returning False, it means a valid sequence of operations
    # exists such that each element `nums[i]` effectively becomes zero when it's processed.
    # The boundary check `if op_i > 0 and i + k > n:` ensures that the trailing elements
    # (last k-1 elements) can also be made zero.
    return True