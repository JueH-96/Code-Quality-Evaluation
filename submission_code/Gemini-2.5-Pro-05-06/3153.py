from typing import List

class Solution:
  def maxSum(self, nums: List[int], k: int) -> int:
    MOD = 10**9 + 7
    
    # Max value of nums[i] is 10^9.
    # 2^29 = 536,870,912
    # 2^30 = 1,073,741,824
    # So, numbers are less than 2^30. Bits 0 through 29 are relevant.
    # This means we need 30 spots in bit_counts.
    num_bits = 30
    bit_counts = [0] * num_bits # bit_counts[j] stores count of numbers having j-th bit set
    
    for x in nums:
      for j in range(num_bits):
        if (x >> j) & 1: # Check if j-th bit of x is set
          bit_counts[j] += 1
          
    # Construct the k largest numbers by distributing the available bits.
    # target_values[i] will be the (i+1)-th largest number we can form.
    # So target_values[0] will be the largest, target_values[1] second largest, etc.
    target_values = [0] * k
    
    # Iterate over bit positions.
    # The order of iterating j (e.g., 0 to 29 vs 29 down to 0) does not change the
    # final multiset of numbers {target_values[0], ..., target_values[k-1]}.
    # The specific loop order used here (outer loop j, inner loop i) ensures that
    # target_values[0] >= target_values[1] >= ... >= target_values[k-1].
    for j in range(num_bits): # For j-th bit (value 2^j)
      # Distribute this bit among the k target numbers if available.
      # target_values[0] gets priority, then target_values[1], and so on.
      for i in range(k): # Iterate over the k numbers we are constructing
        if bit_counts[j] > 0:
          target_values[i] += (1 << j) # Add 2^j to target_values[i]
          bit_counts[j] -= 1          # One j-th bit used up
        else:
          # No more j-th bits left to distribute from bit_counts[j].
          # So target_values[i] (and subsequent target_values[m] for m > i)
          # cannot get this j-th bit. We can break from the inner loop.
          break 
          
    ans = 0
    for val in target_values:
      # Each val is constructed. Its square can be large.
      # For val up to 2^30 - 1 (approx 10^9), val*val can be up to (10^9)^2 = 10^18.
      # Python handles large integers, so direct computation (val * val) is fine.
      # Then take modulo. pow(val, 2, MOD) computes (val*val) % MOD efficiently.
      term_sq = pow(val, 2, MOD)
      ans = (ans + term_sq) % MOD
      
    return ans