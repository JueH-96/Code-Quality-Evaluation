import collections
from typing import List

class Solution:
  def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
    n = len(nums)
    
    ans = 0
    
    # freq stores the counts of (P[idx] % modulo) values encountered so far.
    # P[idx] is the prefix sum of elements equal to 1 in the transformed binary array b.
    # Key: P_val_mod (a value that some P[idx] % modulo equals)
    # Value: count (how many prefix sums P[idx] have resulted in P_val_mod)
    freq = collections.defaultdict(int)
    
    # Base case: The prefix sum before any elements (P[0]) is 0.
    # So, P[0] % modulo = 0. We record that we've seen one prefix sum (P[0])
    # that has a modulo value of 0.
    freq[0] = 1
    
    # current_prefix_sum_val represents the prefix sum of the b array up to the current point.
    # Specifically, after processing nums[r], it stores P[r+1].
    current_prefix_sum_val = 0
    
    for r in range(n): # r is the 0-indexed right endpoint of subarrays nums[l..r]
      # Update current_prefix_sum_val based on nums[r]
      # This is equivalent to: current_prefix_sum_val += b[r]
      if nums[r] % modulo == k:
        current_prefix_sum_val += 1
      # Note: Python booleans True/False convert to 1/0 in arithmetic.
      # So, current_prefix_sum_val += (nums[r] % modulo == k) also works.
      
      # Now, current_prefix_sum_val is P[r+1].
      # We are looking for P[l] such that (P[r+1] - P[l]) % modulo == k.
      # This is equivalent to P[l] % modulo == (P[r+1] - k + modulo) % modulo.
      
      # Calculate the target modulo value for P[l].
      target_mod = (current_prefix_sum_val - k + modulo) % modulo
      
      # Add the count of P[l]'s that satisfy this condition.
      # The `freq` map stores counts for P[0]%modulo, ..., P[r]%modulo.
      # Each P[l] (where P[l]%modulo == target_mod) forms an interesting subarray ending at r.
      ans += freq[target_mod]
      
      # Update freq with the current prefix sum P[r+1] (i.e., current_prefix_sum_val).
      # Specifically, add (current_prefix_sum_val % modulo) to the map.
      freq[current_prefix_sum_val % modulo] += 1
      
    return ans