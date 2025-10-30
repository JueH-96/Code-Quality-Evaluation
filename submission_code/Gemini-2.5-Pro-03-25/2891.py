import collections # Import statement needed for using List type hint, provided implicitly in some environments like LeetCode. Remove if not necessary.
from typing import List # Import List from typing module for type hinting.

class Solution:
  """
  This class provides a solution to find the maximum beauty of an array.
  The beauty is defined as the length of the longest subsequence consisting of equal elements
  after applying an operation at most once to each element.
  The operation allows replacing nums[i] with any integer in [nums[i] - k, nums[i] + k].
  """
  def maximumBeauty(self, nums: List[int], k: int) -> int:
    """
    Finds the maximum possible beauty of the array nums.

    The core idea is based on the observation that a set of numbers `nums[i_1], ..., nums[i_p]` 
    can all be modified to the same target value `x` if and only if the difference between 
    the maximum and minimum values in this set is at most `2*k`. 
    Proof:
    If they can all be modified to `x`, then for each `nums[i_j]`, `nums[i_j] - k <= x <= nums[i_j] + k`.
    This implies `x >= max(nums[i_j]) - k` and `x <= min(nums[i_j]) + k`.
    So, `max(nums[i_j]) - k <= min(nums[i_j]) + k`, which rearranges to `max(nums[i_j]) - min(nums[i_j]) <= 2*k`.
    Conversely, if `max(S) - min(S) <= 2*k` for a subset S, let `min_S = min(S)` and `max_S = max(S)`.
    Then `max_S - k <= min_S + k`. The interval `[max_S - k, min_S + k]` is non-empty. 
    Any `x` in this interval satisfies `max_S - k <= x <= min_S + k`.
    For any element `s` in `S`, we have `min_S <= s <= max_S`.
    So `s - k <= max_S - k <= x` and `x <= min_S + k <= s + k`.
    Thus, `s - k <= x <= s + k` for all `s` in `S`. All elements in S can be transformed to `x`.

    Therefore, the problem reduces to finding the largest subset S of nums such that
    `max(S) - min(S) <= 2*k`. This is equivalent to finding the longest subarray
    in the sorted version of `nums` that satisfies this condition.

    This equivalent problem can be solved efficiently using sorting and a sliding window.

    Args:
      nums: A list of integers. The input array.
      k: A non-negative integer specifying the range for the operation `[value - k, value + k]`.

    Returns:
      The maximum possible beauty of the array, which is the size of the largest subset found.
    """
    
    # Sort the array to enable the sliding window approach.
    # Sorting takes O(N log N) time, where N is the length of nums.
    nums.sort()
    
    n = len(nums)
    # Initialize max_length to 0. For N >= 1, it will be updated to at least 1.
    max_length = 0 
    # Initialize the left pointer of the sliding window.
    left = 0
    
    # The sliding window approach takes O(N) time because both 'left' and 'right' pointers
    # only move forward, each traversing the array at most once in total.
    # 'right' pointer expands the window to the right.
    for right in range(n):
      # The current window contains elements nums[left...right].
      # Since the array is sorted, the minimum element in the window is nums[left].
      # The maximum element in the window is nums[right].
      
      # Check if the condition max(window) - min(window) <= 2*k holds.
      # Condition translates to: nums[right] - nums[left] <= 2 * k
      
      # While the condition is violated (the difference is too large),
      # we must shrink the window from the left by incrementing the 'left' pointer.
      # This process continues until the window satisfies the condition.
      while nums[right] - nums[left] > 2 * k:
        left += 1
        
      # At this point, the window nums[left...right] satisfies the condition nums[right] - nums[left] <= 2*k.
      # Calculate the length of this valid window. The length is (right - left + 1).
      current_length = right - left + 1
      
      # Update the maximum length found so far across all valid windows encountered.
      max_length = max(max_length, current_length)
      
    # After iterating through all possible 'right' positions, max_length holds the maximum size
    # of a subarray satisfying the condition, which corresponds to the maximum beauty.
    return max_length