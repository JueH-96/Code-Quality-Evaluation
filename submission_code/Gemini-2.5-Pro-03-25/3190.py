import math
from typing import List

class Solution:
  """
  Solves the problem of finding the minimum swaps to make the last elements
  of two arrays the maximums of their respective arrays.
  """
  def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
    """
    Calculates the minimum operations (swaps) required to satisfy the conditions:
    1. nums1[n - 1] is the maximum element in nums1.
    2. nums2[n - 1] is the maximum element in nums2.
    An operation consists of swapping nums1[i] and nums2[i] for some index i.

    Args:
        nums1: The first integer array.
        nums2: The second integer array. Both arrays have the same length n.

    Returns:
        The minimum number of operations (swaps), or -1 if it is impossible to satisfy both conditions.
    """
    n = len(nums1)

    # Helper function defined inside to capture nums1, nums2, n from the outer scope.
    # This avoids passing these variables explicitly.
    def calculate_cost(target1: int, target2: int, initial_swaps: int) -> float:
      """
      Calculates the cost (number of swaps) for a given scenario.
      A scenario is determined by the potential values at index n-1 after operations.
      These values, target1 and target2, must become the maximums of their respective arrays.
      The parameter initial_swaps accounts for whether the elements at index n-1 were swapped to achieve these targets.

      Args:
          target1: The value required at nums1[n-1], which must be max(nums1).
          target2: The value required at nums2[n-1], which must be max(nums2).
          initial_swaps: The number of swaps already counted (0 or 1 depending on whether the swap at index n-1 was performed to determine target1 and target2).

      Returns:
          The total minimum number of swaps required for this scenario to satisfy the conditions
          for indices 0 to n-2. Returns float('inf') if the conditions cannot be satisfied for any index i < n-1.
      """
      current_swaps = initial_swaps
      # Iterate through all indices except the last one (n-1).
      # For each index i < n-1, we need to ensure that after potential swap,
      # nums1[i] <= target1 and nums2[i] <= target2.
      for i in range(n - 1):
        a_i = nums1[i]
        b_i = nums2[i]
        
        # Check if the pair (a_i, b_i) satisfies the conditions: a_i <= target1 and b_i <= target2.
        # This corresponds to NOT swapping elements at index i.
        possible_no_swap = (a_i <= target1 and b_i <= target2)
        
        # Check if the swapped pair (b_i, a_i) satisfies the conditions: b_i <= target1 and a_i <= target2.
        # This corresponds to swapping elements at index i.
        possible_with_swap = (b_i <= target1 and a_i <= target2)
        
        # If neither keeping the original pair nor swapping works for index i,
        # then it's impossible to satisfy the conditions for this scenario with the given targets.
        if not possible_no_swap and not possible_with_swap:
          return float('inf')
        
        # If keeping the original pair does not work (possible_no_swap is False),
        # but swapping does work (possible_with_swap must be True because the 'impossible' check above passed),
        # then we are forced to swap at index i to satisfy the conditions. Increment the swap count.
        if not possible_no_swap:
          current_swaps += 1
        # else: (if possible_no_swap is True)
          # If keeping the original pair works, we choose this option as it potentially requires
          # zero swaps for this index. Even if swapping also works,
          # we prefer not swapping to minimize the total operations.
          # No action needed, current_swaps remains unchanged.
          
      # Return the total number of swaps calculated for this scenario.
      # Use float type to handle potential infinity value consistently.
      return float(current_swaps) 

    # Consider two main scenarios based on the choice at index n-1:

    # Scenario 1: Do not swap elements at index n-1.
    # The final values at index n-1 will be nums1[n-1] and nums2[n-1].
    # These must be the maximums, so set target1 = nums1[n-1], target2 = nums2[n-1].
    # Initial swaps count is 0 because no swap occurred at n-1 for this scenario.
    cost1 = calculate_cost(nums1[n-1], nums2[n-1], 0)

    # Scenario 2: Swap elements at index n-1.
    # The final values at index n-1 will be nums2[n-1] and nums1[n-1].
    # These must be the maximums, so set target1 = nums2[n-1], target2 = nums1[n-1].
    # Initial swaps count is 1 because a swap occurred at n-1 for this scenario.
    cost2 = calculate_cost(nums2[n-1], nums1[n-1], 1)

    # The overall minimum operations required is the minimum cost found across the two possible scenarios.
    min_cost = min(cost1, cost2)

    # If min_cost is still infinity, it implies that neither scenario could satisfy the conditions
    # for all indices i < n-1. Therefore, it's impossible to meet the requirements.
    if min_cost == float('inf'):
      return -1
    else:
      # Otherwise, a solution exists, and min_cost holds the minimum number of swaps.
      # Return this value, converted to an integer as required.
      return int(min_cost)