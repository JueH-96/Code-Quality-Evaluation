import math
from typing import List

class Solution:
  def findKthSmallest(self, coins: List[int], k: int) -> int:
    # Step 1: Filter coins
    # Remove c_i if it's a multiple of some other c_j already kept.
    # Sorting coins first ensures that if c_i is a multiple of c_j where c_j < c_i,
    # c_j will be processed and potentially kept before c_i.
    coins.sort()
    
    filtered_coins = []
    for coin_val in coins:
      is_redundant = False
      # Check if coin_val is a multiple of any coin already added to filtered_coins.
      for kept_coin in filtered_coins:
        if coin_val % kept_coin == 0:
          is_redundant = True
          break
      if not is_redundant:
        filtered_coins.append(coin_val)
    
    coins = filtered_coins
    n = len(coins)

    # Helper for LCM.
    # (a // gcd) * b is safer against overflow in non-Python languages, 
    # but Python handles large integers, so (a * b) // gcd is fine.
    def _lcm(a: int, b: int) -> int:
      if a == 0 or b == 0: # Should not happen with coin values >= 1
        return 0
      # Optimization: if current_lcm starts at 1, lcm(1, x) is x.
      # math.gcd(1, x) is 1, so (1 * x) // 1 = x. This explicit check is a micro-optimization.
      if a == 1: return b
      if b == 1: return a
      
      # Calculate LCM. abs() isn't needed as coins are positive.
      res = (a * b) // math.gcd(a, b)
      return res

    # count_le(limit_val) returns the count of distinct numbers <= limit_val
    # that are multiples of at least one coin in `coins`.
    # Uses the Principle of Inclusion-Exclusion.
    def count_le(limit_val: int) -> int:
      count = 0
      # Iterate over all non-empty subsets of coins.
      # A bitmask `mask` from 1 to (1 << n) - 1 represents a subset.
      # (1 << n) is 2^n. So range is [1, 2^n - 1].
      for i in range(1, 1 << n):
        current_lcm_val = 1
        num_elements_in_subset = 0
        
        for j in range(n):
          # Check if the j-th coin is in the current subset (represented by bitmask i)
          if (i >> j) & 1: 
            num_elements_in_subset += 1
            current_lcm_val = _lcm(current_lcm_val, coins[j])
            
            # Optimization: if LCM already exceeds limit_val, it won't contribute.
            # Its term limit_val // current_lcm_val will be 0.
            # Further multiplications will only make current_lcm_val larger.
            if current_lcm_val > limit_val:
              break # Exit early from LCM calculation for this subset
        
        # If LCM calculation was broken early (or naturally became > limit_val after processing all elements)
        if current_lcm_val > limit_val:
            continue # This subset's contribution is 0.

        # Add or subtract term based on subset size (Principle of Inclusion-Exclusion)
        term = limit_val // current_lcm_val
        if num_elements_in_subset % 2 == 1: # Odd number of elements in subset, add
          count += term
        else: # Even number of elements in subset, subtract
          count -= term
      return count

    # Binary search for the k-th smallest amount.
    # Smallest coin in filtered_coins is coins[0] (it's sorted and non-empty as per constraints).
    # Smallest possible answer is coins[0].
    # Largest reasonable upper bound: if only coins[0] was used, k-th amount would be coins[0]*k.
    # The problem constraints ensure coins is not empty and coins[0] >= 1.
    low = 1 
    high = coins[0] * k # Max possible value: 25 * 2*10^9 = 5 * 10^10
    
    ans = high # Initialize ans with a valid upper value that satisfies count >= k

    while low <= high:
      mid = low + (high - low) // 2
      if mid == 0: # Should not happen as low starts at 1. Defensive check.
          low = 1 
          continue
          
      num_multiples = count_le(mid)
      
      if num_multiples >= k:
        # mid might be the k-th smallest, or something larger.
        # Store it as a candidate and try to find a smaller one.
        ans = mid
        high = mid - 1
      else: # num_multiples < k
        # mid is too small, k-th smallest must be larger.
        low = mid + 1
        
    return ans