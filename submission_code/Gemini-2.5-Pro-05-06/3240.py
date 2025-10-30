class Solution:
  def findMaximumNumber(self, k: int, x: int) -> int:
    
    def count_total_price(n_val: int) -> int:
      if n_val == 0:
        return 0
      
      total_price = 0
      # Iterate through bit positions i (1-indexed from right)
      # The i-th bit corresponds to the value 2^(i-1)
      for i in range(1, 65): # Max N is ~10^15, so log2(N) is ~50. 64 is a safe upper limit for bit positions.
        
        p_val = 1 << (i - 1) # Value 2^(i-1)

        # Optimization: if 2^(i-1) > n_val, then no number from 1 to n_val 
        # can have this i-th bit (or any higher bit) set.
        # So, count for this bit position (and higher ones) will be 0.
        if p_val > n_val:
          break 
        
        if i % x == 0:
          # This bit position contributes to the price.
          # Count how many numbers from 1 to n_val have the i-th bit set.
          # This is equivalent to counting numbers in [0, n_val] that have the (i-1)-th (0-indexed) bit set,
          # as 0 doesn't have bits set that affect the count for positive powers of 2.
          
          # The (i-1)-th bit (0-indexed) toggles every p_val numbers.
          # A full cycle of this bit is 2 * p_val numbers (0s then 1s, or vice versa).
          len_cycle = p_val * 2 # This is 1 << i
          
          # Numbers are effectively 0-indexed for this calculation, spanning n_val+1 values (0 to n_val).
          num_count_for_formula = n_val + 1
          
          # Number of full cycles in num_count_for_formula numbers
          num_full_cycles = num_count_for_formula // len_cycle
          
          count_for_this_bit = num_full_cycles * p_val
          
          # Contribution from the remaining part of the numbers
          remainder_len = num_count_for_formula % len_cycle
          count_for_this_bit += max(0, remainder_len - p_val)
          
          total_price += count_for_this_bit
          
      return total_price

    low = 0
    # Max k is 10^15.
    # Max num can be estimated: N log N ~ 2*x*k.
    # If x=8, k=10^15, N log N ~ 16 * 10^15. N is roughly 3*10^14.
    # A high bound of 2*10^15 is generous and safe.
    high = 2 * (10**15) 
    ans = 0

    while low <= high:
      mid = low + (high - low) // 2
      # mid can be 0 if low=0. count_total_price(0) returns 0, which is correct.
      current_sum_prices = count_total_price(mid)
      
      if current_sum_prices <= k:
        ans = mid
        low = mid + 1
      else:
        high = mid - 1
        
    return ans