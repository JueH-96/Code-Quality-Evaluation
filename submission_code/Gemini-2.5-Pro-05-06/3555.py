from typing import List

class Solution:
  def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
    """
    Performs k operations on nums. In each operation:
    1. Find the minimum value x in nums.
    2. If multiple occurrences, select the one that appears first (smallest index).
    3. Replace the selected x with x * multiplier.
    Returns the final state of nums after all k operations.

    Constraints:
    1 <= nums.length <= 100
    1 <= nums[i] <= 100
    1 <= k <= 10
    1 <= multiplier <= 5
    """

    # Based on the constraints:
    # - nums is never empty (nums.length >= 1).
    # - k is at least 1 (k >= 1). This means the loop will run at least once.
    #   If k could be 0 (though constraint says k >= 1), range(0) would correctly 
    #   result in no operations, and the original nums would be returned.

    for _ in range(k):
        # Step 1: Find the minimum value x in nums.
        # The min() function iterates through the list to find the minimum element.
        # Complexity: O(N), where N = len(nums).
        # Since nums.length >= 1 by constraint, min(nums) is always valid.
        min_val = min(nums)

        # Step 2: If there are multiple occurrences of the minimum value,
        # select the one that appears first.
        # The list.index(value) method returns the index of the first occurrence
        # of 'value' in the list.
        # Complexity: O(N) in the worst case.
        # Since min_val was obtained from nums, it is guaranteed to be in the list,
        # so list.index() will not raise a ValueError.
        idx_to_modify = nums.index(min_val)

        # Step 3: Replace the selected minimum value x with x * multiplier.
        # nums[idx_to_modify] currently holds min_val.
        # We update this element in-place.
        nums[idx_to_modify] = nums[idx_to_modify] * multiplier
        # An alternative, equivalent assignment would be:
        # nums[idx_to_modify] = min_val * multiplier

    # After k operations, return the modified list.
    return nums