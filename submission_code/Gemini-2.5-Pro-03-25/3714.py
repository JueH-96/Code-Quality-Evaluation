import sys 
# Setting a higher recursion depth isn't usually needed for this iterative approach, 
# but good practice if there were recursive helper functions.
# sys.setrecursionlimit(2000) 

from typing import List

# Define the modulus value as required by the problem
MOD = 10**9 + 7

class Solution:
  """
  Calculates the sum of (minimum + maximum) elements over all non-empty subsequences 
  of nums with length at most k, modulo 10^9 + 7.
  The approach uses combinatorics and dynamic programming ideas based on sorted distinct elements.
  """
  def minMaxSums(self, nums: List[int], k: int) -> int:
    """
    Args:
        nums: The input integer array. Constraints: 1 <= nums.length <= 10^5.
        k: The maximum length of subsequences to consider. Constraints: 1 <= k <= min(70, nums.length).

    Returns:
        The total sum modulo 10^9 + 7.
    """
    N = len(nums)
    
    # Constraints state N >= 1, k >= 1. 
    # No need to handle N=0 or k=0 explicitly, as they are ruled out by constraints.
    
    # Precompute factorials and inverse factorials modulo MOD
    # This is needed for efficient computation of combinations (nCr)
    fact = [1] * (N + 1)
    invfact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    
    # Compute inverse of N! using Fermat's Little Theorem: a^(MOD-2) % MOD is a^-1 % MOD for prime MOD
    # This requires MOD to be prime, which 10^9 + 7 is.
    invfact[N] = pow(fact[N], MOD - 2, MOD)
    
    # Compute inverse factorials iteratively downwards using relation: (i!)⁻¹ = ((i+1)!)⁻¹ * (i+1)
    for i in range(N - 1, -1, -1):
        invfact[i] = (invfact[i + 1] * (i + 1)) % MOD

    # Helper function to compute nCr mod MOD using precomputed values
    def nCr_mod(n, r):
        """ Computes nCr modulo MOD efficiently """
        if r < 0 or r > n:
            # If r is out of bounds [0, n], the combination is 0
            return 0
        # Base cases for combinations
        if r == 0 or r == n:
             # C(n, 0) = 1, C(n, n) = 1
             return 1
        # Optimization: C(n, r) = C(n, n-r). Use smaller index for potentially faster factorial lookup?
        # Actually, with precomputed factorials and inverse factorials, complexity is O(1) regardless.
        # Still, using smaller 'r' might be slightly numerically better in some contexts, but not critical here.
        if r > n // 2:
             r = n - r 
             
        # Formula: nCr = n! / (r! * (n-r)!)
        # Modulo arithmetic: nCr mod M = (n! * (r!)⁻¹ * ((n-r)!)⁻¹) mod M
        num = fact[n]
        den = (invfact[r] * invfact[n - r]) % MOD
        return (num * den) % MOD

    # Sort the input array. This allows grouping identical elements and simplifies count logic.
    nums.sort()
    
    # Find distinct values and their frequencies (counts)
    # This step reduces the problem size from N to D distinct elements.
    distinct_nums = []
    counts = []
    if N > 0: # Check N > 0 though guaranteed by constraints
        distinct_nums.append(nums[0])
        counts.append(1)
        for i in range(1, N):
            if nums[i] == nums[i-1]:
                counts[-1] += 1 # Increment count for the last distinct number
            else:
                # Found a new distinct number
                distinct_nums.append(nums[i])
                counts.append(1)
    
    D = len(distinct_nums) # Number of distinct elements

    # Calculate cumulative counts N_j (number of elements <= v_j)
    # N_j: count of elements in 'nums' that are less than or equal to the j-th distinct value v_j
    cumulative_counts_le = []
    current_le_count = 0
    for count in counts:
        current_le_count += count
        cumulative_counts_le.append(current_le_count)
    
    # Calculate the total sum of maximum elements over all valid subsequences
    sum_max = 0
    # T(v_j) = sum_{s=1..k} C(N_j, s) is the number of non-empty subsequences of length <= k using elements <= v_j
    T_prev = 0 # T(v_0) = 0, representing count for elements strictly less than v_1 (which is an empty set)
    
    # Iterate through distinct elements v_1, v_2, ..., v_D
    for j in range(D):
        v_j = distinct_nums[j] # The j-th distinct value
        N_j = cumulative_counts_le[j] # Total count of elements <= v_j
        
        # Compute T(v_j) = sum_{s=1}^k C(N_j, s) mod M
        T_curr = 0
        for s in range(1, k + 1):
            if s > N_j: # Optimization: C(N_j, s) = 0 if s > N_j. Can stop summing early.
                break
            T_curr = (T_curr + nCr_mod(N_j, s)) % MOD
        
        # Number of subsequences with maximum value exactly v_j is T(v_j) - T(v_{j-1})
        count_max_vj = (T_curr - T_prev + MOD) % MOD # Add MOD before modulo to handle potential negative result
        # Add the contribution: v_j * count_max_vj
        # Take v_j % MOD defensively, though nums[i] <= 10^9 < MOD.
        sum_max = (sum_max + (v_j % MOD) * count_max_vj) % MOD 
        
        T_prev = T_curr # Update T_prev for the next iteration (for v_{j+1})

    # Calculate cumulative counts N'_j (number of elements >= v_j)
    # N'_j: count of elements in 'nums' that are greater than or equal to the j-th distinct value v_j
    cumulative_counts_ge = []
    current_ge_count = N # Start with total N elements
    for j in range(D):
         # N'_j is the count of elements >= v_j
         cumulative_counts_ge.append(current_ge_count)
         # For the next distinct value v_{j+1}, elements equal to v_j are excluded.
         # So subtract count of v_j from current_ge_count.
         current_ge_count -= counts[j] 
         
    # Calculate the total sum of minimum elements over all valid subsequences
    sum_min = 0
    # T'(v_j) = sum_{s=1..k} C(N'_j, s) is the number of non-empty subsequences of length <= k using elements >= v_j
    T_prime_next = 0 # T'(v_{D+1}) = 0, representing count for elements strictly greater than v_D (empty set)
    
    # Iterate backwards through distinct elements (from v_D down to v_1)
    for j in range(D - 1, -1, -1):
         v_j = distinct_nums[j] # The j-th distinct value
         N_prime_j = cumulative_counts_ge[j] # Total count of elements >= v_j

         # Compute T'(v_j) = sum_{s=1}^k C(N'_j, s) mod M
         T_prime_curr = 0
         for s in range(1, k + 1):
             if s > N_prime_j: # Optimization: C(N'_j, s) = 0 if s > N'_j. Stop summing early.
                 break
             T_prime_curr = (T_prime_curr + nCr_mod(N_prime_j, s)) % MOD
         
         # Number of subsequences with minimum value exactly v_j is T'(v_j) - T'(v_{j+1})
         count_min_vj = (T_prime_curr - T_prime_next + MOD) % MOD # Add MOD before modulo for safety
         # Add the contribution: v_j * count_min_vj
         sum_min = (sum_min + (v_j % MOD) * count_min_vj) % MOD # Take v_j % MOD defensively
         
         # Update T_prime_next for the next iteration (which processes v_{j-1})
         # It will need T'(v_j) to compute T'(v_{j-1}) - T'(v_j)
         T_prime_next = T_prime_curr
         
    # The final answer is the sum of the total maximums and total minimums, modulo MOD
    total_sum = (sum_max + sum_min) % MOD
    return total_sum