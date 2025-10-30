import collections
from typing import List

class Solution:
  def isGood(self, nums: List[int]) -> bool:
    # Constraints: 1 <= nums.length <= 100, 1 <= nums[i] <= 200.
    # These ensure nums is not empty and elements are positive.
    
    # Step 1: Determine the candidate for n.
    # If nums is a permutation of base[k], then max(nums) must be k.
    # Let n_val be this candidate k.
    n_val = max(nums)

    # Step 2: Check Length.
    # The length of base[n_val] is n_val + 1.
    # nums must have this length.
    if len(nums) != n_val + 1:
      return False

    # Step 3: Check Frequencies.
    # We use collections.Counter to get frequencies of numbers in nums.
    counts = collections.Counter(nums)

    # Check frequencies for numbers from 1 to n_val-1. Each must appear once.
    # If n_val is 1, base[1] = [1,1]. The loop 'for i in range(1, 1)' is empty, which is correct.
    # In this case (n_val=1), only the check for counts[n_val] (i.e. counts[1]) runs.
    for i in range(1, n_val):
      if counts[i] != 1:
        # A collections.Counter returns 0 for keys not present.
        # If number i is missing (counts[i] == 0) or has a different count, it's not good.
        return False
    
    # Check frequency for the number n_val itself. It must appear twice.
    if counts[n_val] != 2:
      # If n_val's count is not 2, it's not good.
      # (counts[n_val] cannot be 0 here because n_val = max(nums), so it must be in nums).
      return False
        
    # If all conditions are met:
    # - Length is correct (n_val + 1).
    # - n_val is the maximum element.
    # - All numbers 1..n_val-1 appear once.
    # - n_val appears twice.
    # - All elements are >= 1 (from constraints).
    # - The total count of elements verified is (n_val-1)*1 + 2 = n_val+1, which equals len(nums).
    #   This ensures no other numbers or incorrect duplicates exist.
    return True