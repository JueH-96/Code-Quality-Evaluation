from typing import List

class Solution:
  def sumOfPower(self, nums: List[int], k: int) -> int:
    n = len(nums)
    MOD = 10**9 + 7

    # Precompute powers of 2 modulo MOD
    # pow2[i] will store 2^i % MOD
    # We need powers up to 2^n
    pow2 = [1] * (n + 1) # pow2[0] = 2^0 = 1
    for i in range(1, n + 1):
      pow2[i] = (pow2[i - 1] * 2) % MOD

    # dp[s][l] = number of subsequences of nums (processed so far)
    #            that sum to 's' and have length 'l'.
    # Dimensions: (k+1) for sum (0 to k), (n+1) for length (0 to n).
    dp = [[0] * (n + 1) for _ in range(k + 1)]
    
    # Base case: The empty subsequence has sum 0 and length 0.
    # There is one such subsequence.
    dp[0][0] = 1

    # Iterate through each number in nums
    for x_val in nums:
      # Iterate sum 's_curr' downwards. 's_curr' is the sum *after* including x_val.
      # So, s_curr must be at least x_val.
      # The sum *before* including x_val was (s_curr - x_val).
      for s_curr in range(k, x_val - 1, -1): 
        # Iterate length 'l_curr' downwards. 'l_curr' is the length *after* including x_val.
        # So, l_curr must be at least 1 (as x_val is included).
        # The length *before* including x_val was (l_curr - 1).
        for l_curr in range(n, 0, -1): 
          # If dp[s_curr - x_val][l_curr - 1] is non-zero, it means there are
          # subsequences summing to (s_curr - x_val) with length (l_curr - 1)
          # formed from elements processed *before* the current x_val.
          # We add x_val to these subsequences.
          # The condition `> 0` is a small optimization; adding 0 changes nothing.
          if dp[s_curr - x_val][l_curr - 1] > 0:
            dp[s_curr][l_curr] = (dp[s_curr][l_curr] + dp[s_curr - x_val][l_curr - 1]) % MOD
            
    total_sum_of_power = 0
    # After filling the dp table using all numbers in nums:
    # dp[k][l_S_prime] is the count of subsequences S' of nums
    # that sum to k and have length l_S_prime.
    
    # Iterate over all possible lengths 'l_S_prime' for these subsequences S'
    for l_S_prime in range(n + 1): # Length can range from 0 to n.
                                   # (Effectively 1 to n, since k > 0, so l_S_prime cannot be 0 for sum k)
      if dp[k][l_S_prime] > 0:
        count_S_prime = dp[k][l_S_prime]
        
        # Each such S' (subsequence of nums, sum k, length l_S_prime)
        # contributes 2^(n - l_S_prime) to the total sum of powers.
        # This is because S' must be a subsequence of S (a subsequence of nums whose power we calculate),
        # and S is a subsequence of nums.
        # For a fixed S', S must contain all elements of S'.
        # The remaining (n - l_S_prime) elements of nums (those not in S') 
        # can either be chosen to be in S or not. There are 2^(n - l_S_prime) ways.
        num_supersets_S_containing_S_prime = pow2[n - l_S_prime]
        
        term = (count_S_prime * num_supersets_S_containing_S_prime) % MOD
        total_sum_of_power = (total_sum_of_power + term) % MOD
        
    return total_sum_of_power