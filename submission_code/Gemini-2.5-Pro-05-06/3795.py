import collections
from typing import List

class Solution:
  def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
    n = len(nums)
    m = len(queries)

    def check(k_param: int) -> bool:
      # This function determines if nums can be made a zero array using the first k_param queries.
      
      # Base case: k_param = 0 means no queries are used.
      # If nums is already all zeros, then 0 queries are needed.
      if k_param == 0:
        for x_val in nums:
          if x_val != 0:
            return False
        return True

      # For each nums[idx], check if its value can be reduced to 0
      for idx in range(n):
        target_sum = nums[idx]
        if target_sum == 0: # Already zero, no operations needed for this element
          continue

        # Collect applicable val_i from the first k_param queries for nums[idx]
        # and count occurrences of each val_i (from 1 to 10)
        counts_for_val = collections.defaultdict(int)
        # Iterate through the first k_param queries (indices 0 to k_param-1)
        for i in range(k_param): 
          l_i, r_i, val_i = queries[i]
          if l_i <= idx <= r_i: # Check if query is applicable to current nums[idx]
            counts_for_val[val_i] += 1
        
        # Subset sum DP: can target_sum be formed by collected vals?
        # dp[s] is True if sum s can be formed.
        dp = [False] * (target_sum + 1)
        dp[0] = True # Sum of 0 is always possible (by choosing no coins)

        for val_coin in range(1, 11): # Iterate through coin denominations 1 to 10
          if counts_for_val[val_coin] == 0: # No coins of this denomination available
            continue
          
          num_available_coins_of_this_val = counts_for_val[val_coin]
          
          # Use binary decomposition (aka "powers of 2" trick) for bounded knapsack items.
          # This converts a set of `C` identical items into `log C` distinct "meta-items".
          power_of_2 = 1
          while num_available_coins_of_this_val > 0:
            # Number of actual coins of value 'val_coin' to bundle in this 'meta-coin'
            batch_size_in_num_coins = min(power_of_2, num_available_coins_of_this_val)
            # Value of the current 'meta-coin'
            meta_coin_value = batch_size_in_num_coins * val_coin

            # Standard knapsack update: iterate sums downwards
            # s ranges from target_sum down to meta_coin_value
            for s in range(target_sum, meta_coin_value - 1, -1):
              if dp[s - meta_coin_value]: # If sum `s - meta_coin_value` was possible
                dp[s] = True # Then sum `s` is also possible by adding this meta-coin
            
            num_available_coins_of_this_val -= batch_size_in_num_coins
            power_of_2 *= 2 # Next power of 2 for batching
            
        if not dp[target_sum]:
          # nums[idx] cannot be made zero with the first k_param queries
          return False 

      # All nums[i] can be made zero
      return True 

    min_k_found = -1
    # k can range from 0 (no queries) to m (all m queries)
    low = 0
    high = m 

    while low <= high:
      mid_k = low + (high - low) // 2
      if check(mid_k):
        min_k_found = mid_k
        high = mid_k - 1 # Try to find an even smaller k
      else:
        low = mid_k + 1 # k_param is too small, need more queries
        
    return min_k_found