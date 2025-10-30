import collections

class Solution:
  def maximumSum(self, nums: list[int]) -> int:
    n = len(nums)

    # Step 1: Precompute cores for all indices from 1 to N.
    # cores_list[idx] will store the square-free part of idx (1-based).
    # Size n+1 for 1-based indexing up to n. cores_list[0] is unused.
    cores_list = [0] * (n + 1) 
    
    # Smallest Prime Factor (SPF) array.
    # spf[k] stores the smallest prime factor of k.
    spf = list(range(n + 1)) # spf[i] = i initially for i >= 0.

    # Sieve to calculate SPFs for numbers up to N.
    # Iterate i from 2 up to n.
    # If spf[i] is still i, then i is prime.
    # For all multiples j of i (starting from i*i, as smaller multiples are already covered by smaller primes),
    # if spf[j] is still j, it means j's smallest prime factor found so far is j itself.
    # So, set spf[j] = i.
    for i in range(2, n + 1):
        if spf[i] == i: # i is prime
            for j in range(i * i, n + 1, i): 
                if spf[j] == j: # Check if spf[j] hasn't been set by a smaller prime factor
                    spf[j] = i
    
    # Calculate core for each index i from 1 to n.
    if n >= 1: # nums is guaranteed to have at least one element.
        cores_list[1] = 1 # Core of 1 is 1.
    
    for i in range(2, n + 1):
        # To calculate core(i):
        # Prime factorize i: i = p1^a1 * p2^a2 * ...
        # core(i) is the product of prime factors pj for which exponent aj is odd.
        
        temp_i = i      # Number to factorize to get core
        current_core = 1
        while temp_i > 1:
            prime_factor = spf[temp_i]
            count = 0
            # Count occurrences of this prime_factor
            while temp_i > 0 and temp_i % prime_factor == 0: # temp_i > 0 is technically redundant as prime_factor >= 2
                count += 1
                temp_i //= prime_factor
            
            if count % 2 == 1: # If exponent is odd, include this prime_factor in the core
                current_core *= prime_factor
        cores_list[i] = current_core

    # Step 2: Group nums values by the core of their 1-based index and sum them up.
    sums_by_core = collections.defaultdict(int)
    max_overall_sum = 0

    for i in range(1, n + 1): # Iterate through 1-based indices
        # nums is 0-indexed in Python, so nums[i-1] corresponds to 1-based index i.
        value_for_index_i = nums[i-1] 
        
        core_of_index = cores_list[i]
        
        sums_by_core[core_of_index] += value_for_index_i
        
        # Update maximum sum found so far.
        # Since nums[i] >= 1, sums are always positive and non-decreasing for a given core.
        if sums_by_core[core_of_index] > max_overall_sum:
            max_overall_sum = sums_by_core[core_of_index]
            
    return max_overall_sum