import math
from typing import List

class Solution:
  """
  Solves the problem of counting partitions with an even sum difference.
  """
  def countPartitions(self, nums: List[int]) -> int:
    """
    Calculates the number of partitions where the difference between the sum of the left
    and right subarrays is even.

    A partition is defined by an index i (0 <= i < n-1), splitting the array into
    left subarray nums[0...i] and right subarray nums[i+1...n-1]. Both subarrays
    must be non-empty. The condition is that sum(left) - sum(right) is even.

    Args:
        nums: The input integer array. Length n >= 2.

    Returns:
        The number of valid partitions.
    """
    n = len(nums)

    # Calculate the total sum of the array elements.
    total_sum = sum(nums)

    # Let S_left(i) be the sum of the left subarray nums[0...i].
    # Let S_right(i) be the sum of the right subarray nums[i+1...n-1].
    # We know that S_left(i) + S_right(i) = total_sum for any partition i.
    #
    # We are interested in the number of partitions i (where 0 <= i < n-1) such that
    # the difference D(i) = S_left(i) - S_right(i) is even.
    #
    # We can express the difference in terms of S_left(i) and total_sum:
    # D(i) = S_left(i) - S_right(i)
    # D(i) = S_left(i) - (total_sum - S_left(i))
    # D(i) = S_left(i) - total_sum + S_left(i)
    # D(i) = 2 * S_left(i) - total_sum
    #
    # Now, we need to determine when D(i) is even.
    # The term 2 * S_left(i) is always even, regardless of the value of S_left(i).
    # Let's consider the parity (even/odd) of D(i):
    # D(i) = even_number - total_sum
    #
    # Case 1: total_sum is even.
    #   D(i) = even_number - even_number = even_number.
    #   So, if total_sum is even, the difference D(i) is always even for any partition i.
    #
    # Case 2: total_sum is odd.
    #   D(i) = even_number - odd_number = odd_number.
    #   So, if total_sum is odd, the difference D(i) is always odd for any partition i.
    #
    # Therefore, the condition that the difference is even (D(i) is even) is met if and only if
    # the total sum of the array (total_sum) is even.

    # Check the parity of the total sum.
    if total_sum % 2 == 0:
      # If total_sum is even, then *every* possible partition i (from 0 to n-2)
      # will result in an even difference.
      # The number of possible valid partitions is the number of possible values for i.
      # The index i can range from 0 to n-2 (inclusive), as required by the definition
      # 0 <= i < n - 1.
      # The number of integers in the range [0, n-2] is (n-2) - 0 + 1 = n - 1.
      # Since the constraint is n >= 2, n - 1 is always at least 1.
      return n - 1
    else:
      # If total_sum is odd, then *no* partition i will result in an even difference.
      # The difference D(i) will always be odd.
      # Therefore, the number of partitions satisfying the condition is 0.
      return 0