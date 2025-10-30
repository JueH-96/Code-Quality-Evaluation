from typing import List

class Solution:
  def maximumOr(self, nums: List[int], k: int) -> int:
    n = len(nums)

    # Constraints state 1 <= nums.length, so n >= 1.
    # No need to handle n=0 explicitly based on constraints.

    # prefix_or[i] stores nums[0] | nums[1] | ... | nums[i]
    prefix_or = [0] * n
    prefix_or[0] = nums[0]
    for i in range(1, n):
      prefix_or[i] = prefix_or[i-1] | nums[i]

    # suffix_or[i] stores nums[i] | nums[i+1] | ... | nums[n-1]
    suffix_or = [0] * n
    suffix_or[n-1] = nums[n-1]
    for i in range(n-2, -1, -1): # Iterates i from n-2 down to 0
      suffix_or[i] = suffix_or[i+1] | nums[i]
    
    max_val = 0
    
    # Iterate through each element nums[j], assuming it's the one 
    # we apply all k operations (multiplication by 2^k) to.
    for j in range(n):
      # The element nums[j] is multiplied by 2^k.
      # This is equivalent to nums[j] << k.
      modified_num_j = nums[j] << k

      # Calculate the OR sum of all other elements (which are not modified).
      # These are elements nums[0]...nums[j-1] and nums[j+1]...nums[n-1].
      or_of_other_elements = 0
      
      # OR of elements to the left of nums[j]: nums[0] | ... | nums[j-1]
      # This is prefix_or[j-1]. This part exists only if j > 0.
      if j > 0:
        or_of_other_elements |= prefix_or[j-1]
      
      # OR of elements to the right of nums[j]: nums[j+1] | ... | nums[n-1]
      # This is suffix_or[j+1]. This part exists only if j < n - 1.
      if j < n - 1:
        or_of_other_elements |= suffix_or[j+1]
      
      # The total OR sum if nums[j] is chosen for multiplication.
      # Note: if n=1, j=0. Both if conditions (j>0 and j<n-1) are false.
      # or_of_other_elements remains 0, which is correct.
      current_total_or = or_of_other_elements | modified_num_j
      
      if current_total_or > max_val:
        max_val = current_total_or
        
    return max_val