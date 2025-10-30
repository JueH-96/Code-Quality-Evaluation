from typing import List

class Solution:
  def specialPerm(self, nums: List[int]) -> int:
    n = len(nums)
    MOD = 10**9 + 7

    # dp[mask][i] stores the number of special permutations of elements in 'mask'
    # that end with nums[i].
    # 'mask' is a bitmask representing the subset of nums used.
    # 'i' is the index of the last element in the permutation (0 to n-1).
    dp = [[0] * n for _ in range(1 << n)] # dp[mask][last_element_idx]

    # Base cases: Permutations of length 1.
    # For each nums[i], the permutation [nums[i]] is a valid start.
    # It uses elements represented by mask (1 << i) and ends with nums[i].
    for i in range(n):
      dp[1 << i][i] = 1
    
    # Iterate over masks. When calculating for 'mask', we use its value to update
    # 'next_mask'. Since 'next_mask' will always be numerically greater than 'mask',
    # this order of iteration ensures that dp[mask][i] is fully computed before use.
    for mask in range(1, 1 << n):
      # 'i' is the index of the last element in a permutation using elements from 'mask'.
      for i in range(n):
        # Check if nums[i] is actually in the current 'mask'.
        # If not, nums[i] cannot be the last element of a permutation using elements in 'mask'.
        if not ((mask >> i) & 1):
          continue
        
        # If dp[mask][i] is 0, it means no special permutation of elements in 'mask'
        # ends with nums[i]. So, no paths to extend from this state.
        if dp[mask][i] == 0:
          continue

        # Try to append nums[j] to the permutation.
        # 'j' is the index of the new element to be appended.
        for j in range(n):
          # nums[j] must not already be in 'mask'.
          if not ((mask >> j) & 1): # If (mask >> j) & 1 is 0, nums[j] is not in mask.
            # Check the "special" condition: nums[i] (current last element) and 
            # nums[j] (new element to be appended) must be divisible by one another.
            if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
              next_mask = mask | (1 << j)
              # Add the count of permutations ending in nums[i] (using 'mask' elements)
              # to the count of permutations ending in nums[j] (using 'next_mask' elements).
              dp[next_mask][j] = (dp[next_mask][j] + dp[mask][i]) % MOD
                            
    # Sum up counts for all permutations of length n (using all elements).
    # 'full_mask' is the mask where all bits are set (all numbers used).
    full_mask = (1 << n) - 1
    total_special_perms = 0
    for i in range(n): # The permutation can end with any nums[i].
      total_special_perms = (total_special_perms + dp[full_mask][i]) % MOD
      
    return total_special_perms