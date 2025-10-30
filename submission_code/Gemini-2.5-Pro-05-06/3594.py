import collections
from typing import List

class Solution:
  def getLargestOutlier(self, nums: List[int]) -> int:
    """
    Finds the largest potential outlier in an array nums.
    The array nums has n elements.
    - n-2 elements are "special numbers".
    - One element is the sum of these special numbers (S_special).
    - One element is an "outlier" (O).
    The sum of all elements in nums is S_total = (sum of special numbers) + S_special + O
                                        = S_special + S_special + O
                                        = 2 * S_special + O.
    From this, S_special = (S_total - O) / 2.
    
    The algorithm iterates through each unique value in nums, considering it as a
    potential outlier O (val_k).
    For each val_k:
    1. Calculate the required sum of special numbers: S_special_target = (S_total - val_k) / 2.
    2. If (S_total - val_k) is odd, S_special_target is not an integer, so val_k cannot
       be the outlier with this configuration. Skip.
    3. Check if S_special_target exists as an element in nums, different from the
       element chosen as val_k for the outlier. (Distinct indices requirement)
       - If S_special_target == val_k: We need at least two occurrences of val_k in nums
         (one for outlier, one for S_special). Check `counts[val_k] >= 2`.
       - If S_special_target != val_k: We need at least one occurrence of S_special_target
         in nums. Check `counts[S_special_target] >= 1`. This means `S_special_target` is present in `nums`.
         An element `nums[j]` can be chosen as `S_special_target`. If `nums[i]` was chosen as `val_k` (outlier),
         then since `val_k != S_special_target`, `nums[i] != nums[j]`, which ensures $i 
eq j$.
    4. If such an S_special_target is found (satisfying distinct index rule), val_k is a valid potential outlier. Update
       max_outlier if val_k is larger.
    """
    
    S_total = sum(nums)
    counts = collections.Counter(nums)
    
    # Initialize max_outlier to a value smaller than any possible element.
    # nums[i] can be as low as -1000.
    # The problem guarantees at least one potential outlier exists, so this will be updated.
    max_outlier = -1001 # A value more negative than -1000, e.g., float('-inf') also works.
    
    # Iterate through each unique number `val_k` present in `nums`.
    # `val_k` is a candidate value for the outlier `O`.
    # By iterating `counts` (which iterates keys), we consider each unique value once.
    for val_k in counts: # This iterates through the keys of the counts dictionary
      
      # If `val_k` is the outlier `O`, then `2 * S_special = S_total - val_k`.
      # Let `sum_val_k_diff = S_total - val_k`.
      sum_val_k_diff = S_total - val_k
      
      # `S_special` must be an integer. So, `sum_val_k_diff` must be an even number.
      if sum_val_k_diff % 2 != 0:
        continue # If odd, S_special would not be an integer.
      
      S_special_target = sum_val_k_diff // 2 # Integer division
      
      # Now, we need to check if `S_special_target` (the required sum element value)
      # exists in `nums` and can be chosen such that its index is distinct from the outlier's index.
      
      # Case 1: The required sum element `S_special_target` has the same value as `val_k`.
      if S_special_target == val_k:
        # We need at least two occurrences of `val_k` in `nums`.
        # One `val_k` (e.g., `nums[i]`) serves as the outlier `O`.
        # Another `val_k` (e.g., `nums[j]`) serves as the sum element `S_special`.
        # `counts[val_k] >= 2` ensures that at least two such elements exist, so we can pick `i != j`.
        if counts[val_k] >= 2:
          # `val_k` is a potential outlier value.
          if val_k > max_outlier:
            max_outlier = val_k
      # Case 2: The required sum element `S_special_target` has a different value from `val_k`.
      else:
        # We need at least one occurrence of `S_special_target` in `nums`.
        # Let `nums[i]` be the outlier (value `val_k`).
        # Let `nums[j]` be the sum element (value `S_special_target`).
        # Since `val_k != S_special_target`, `nums[i] != nums[j]`. This automatically implies `i != j`.
        # So, we just need to check if `S_special_target` is present in `nums` at all.
        if counts.get(S_special_target, 0) >= 1: # Or `S_special_target in counts`
          # `val_k` is a potential outlier value.
          if val_k > max_outlier:
            max_outlier = val_k
            
    return max_outlier