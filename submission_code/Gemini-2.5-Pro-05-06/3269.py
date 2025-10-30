from typing import List

class Solution:
  def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
    n = len(nums)
    m = len(pattern)
    
    # Constraints analysis:
    # 2 <= n <= 100
    # 1 <= m < n
    # These imply n >= 2, m >= 1, and m <= n-1.
    # Thus, m+1 <= n, meaning a subarray of length m+1 can always be formed.
    # Also, n-m >= 1. The outer loop range `range(n-m)` will iterate at least once.
    
    count = 0
    
    # The outer loop iterates through possible starting indices `i` for subarrays of length m+1.
    # A subarray `nums[i...i+m]` starts at index `i`.
    # `i` can range from 0 to `n-(m+1) = n-m-1`.
    # The loop `for i in range(n-m)` correctly iterates `i` from `0` to `n-m-1`.
    # This loop will run `n-m` times.
    for i in range(n - m):
      is_match_for_current_i = True # Assume the subarray starting at i matches
      
      # The inner loop checks each of the m conditions defined by the pattern.
      # `k` is the index for `pattern`.
      # `pattern[k]` describes the required relationship between `nums[i+k]` and `nums[i+k+1]`.
      for k in range(m):
        # Get the two numbers from nums to compare
        num_left = nums[i+k]
        num_right = nums[i+k+1]
        
        # Check if the actual relationship matches the pattern's requirement
        expected_relation_code = pattern[k]
        
        if expected_relation_code == 1:
          if not (num_right > num_left):
            is_match_for_current_i = False
            break # Mismatch found, current subarray nums[i...i+m] doesn't match
        elif expected_relation_code == 0:
          if not (num_right == num_left):
            is_match_for_current_i = False
            break # Mismatch found
        elif expected_relation_code == -1: # This is the only other case for pattern[k] values
          if not (num_right < num_left):
            is_match_for_current_i = False
            break # Mismatch found
            
      if is_match_for_current_i:
        # All m conditions of the pattern were met for the subarray starting at i
        count += 1
        
    return count