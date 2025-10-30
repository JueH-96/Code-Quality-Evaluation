from typing import List

class Solution:
  def minOperations(self, nums: List[int], target: int) -> int:
    total_sum = 0
    for x in nums:
      total_sum += x
      
    if total_sum < target:
      return -1

    # Max initial power in nums is 2^30.
    # Max sum of nums is 1000 * 2^30. log2(1000 * 2^30) approx 39.96.
    # So, powers up to 2^39 (index 39) might be formed by aggregation.
    # Target is < 2^31, so target bits are relevant up to index 30 (for 2^30).
    # We need counts array to go up to an index handling powers around 2^39.
    # Let's use 45 as a safe upper limit for power indices.
    # Array size 46 means indices 0 to 45, covering powers 2^0 to 2^45.
    COUNT_ARRAY_MAX_INDEX = 45 
    # counts[j] stores the number of available 2^j values
    counts = [0] * (COUNT_ARRAY_MAX_INDEX + 1) 
    
    for x in nums:
      # x is a power of 2, 1 <= x <= 2^30
      # x.bit_length() - 1 gives p such that x = 2^p.
      # E.g., (1).bit_length()-1 = 0, (2).bit_length()-1 = 1, (8).bit_length()-1 = 3.
      power_idx = x.bit_length() - 1
      # power_idx will be between 0 and 30.
      counts[power_idx] += 1

    operations = 0
    
    # Iterate p from 0 up to COUNT_ARRAY_MAX_INDEX
    # This processes powers 2^0, 2^1, ..., 2^COUNT_ARRAY_MAX_INDEX
    for p in range(COUNT_ARRAY_MAX_INDEX + 1):
      # If target requires the current power 2^p
      # Note: (target >> p) & 1 will be 0 if p > 30, because target < 2^31
      if (target >> p) & 1: # True if p-th bit of target is 1
        if counts[p] > 0:
          counts[p] -= 1 # Use one available 2^p
        else:
          # Need 2^p, but counts[p] is 0.
          # Find the smallest available 2^k (k > p) to break down.
          found_k_idx = -1
          # Search for k_idx from p + 1 up to COUNT_ARRAY_MAX_INDEX
          for k_idx in range(p + 1, COUNT_ARRAY_MAX_INDEX + 1):
            if counts[k_idx] > 0:
              found_k_idx = k_idx
              break
          
          if found_k_idx == -1:
            # Cannot find any 2^k (k > p) to break down.
            # This means it's impossible to form the required 2^p,
            # because all available powers are smaller than 2^p (already processed)
            # or larger powers don't exist.
            return -1 
            
          # Found 2^found_k_idx. Break it down to get 2^p.
          # Cost is (found_k_idx - p) operations.
          operations += (found_k_idx - p)
          counts[found_k_idx] -= 1 # One 2^found_k_idx is used up.
          
          # For each step of breakdown 2^j -> 2*2^(j-1), one 2^(j-1) is kept (added to counts),
          # and the other 2^(j-1) continues to be broken down.
          # This means we add one to counts for each power from found_k_idx-1 down to p.
          for intermediate_p_idx in range(found_k_idx - 1, p - 1, -1): # Down to p (inclusive for p)
            counts[intermediate_p_idx] += 1
          
          # After breakdown, counts[p] increased by 1. Now use this 2^p for target.
          counts[p] -= 1 # This makes counts[p] zero as it was 0, became 1, now 0.
      
      # Pass any remaining 2^p pairs to form 2^(p+1) (carry-over)
      # This must be done regardless of whether target needed 2^p or not.
      if p < COUNT_ARRAY_MAX_INDEX: # Ensure we don't write out of bounds for counts[p+1]
        counts[p+1] += counts[p] // 2
      # else: p == COUNT_ARRAY_MAX_INDEX. This is the last power. No further carry.
      # If counts[COUNT_ARRAY_MAX_INDEX] has remaining items that could sum up to more than target,
      # it does not matter as we only care about satisfying target bits.
      # If target required a power > 2^COUNT_ARRAY_MAX_INDEX, it would have been caught
      # by initial total_sum check or the found_k_idx == -1 check.

    # If the loop completes, all necessary bits of target were satisfied.
    return operations