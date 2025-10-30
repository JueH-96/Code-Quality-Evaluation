import math
from typing import List
from collections import defaultdict

class Solution:
    """
    Finds the number of pairs of non-empty disjoint subsequences (seq1, seq2) 
    of nums such that the Greatest Common Divisor (GCD) of elements in seq1 
    is equal to the GCD of elements in seq2. The result is returned modulo 10^9 + 7.
    """
    def subsequencePairCount(self, nums: List[int]) -> int:
        """
        Calculates the count of subsequence pairs based on GCD equality and disjointness.

        Args:
            nums: A list of integers.

        Returns:
            The total number of such pairs modulo 10^9 + 7.
        """
        MOD = 10**9 + 7
        N = len(nums)
        M = 0
        if not nums: 
            return 0
        
        # Find the maximum value in nums to determine the range of possible GCDs.
        # Any GCD must be less than or equal to the maximum value in the array.
        for x in nums:
            M = max(M, x)
        
        # Precompute for each potential GCD value `g` (from 1 to M), the list of indices `i` 
        # such that `nums[i]` is divisible by `g`. This forms the set S_g for each g.
        # Store this in a dictionary `indices_by_g`.
        indices_by_g = defaultdict(list)
        for i in range(N):
             num = nums[i]
             # Iterate through all divisors of num efficiently using sqrt decomposition.
             for g in range(1, int(math.sqrt(num)) + 1):
                 if num % g == 0:
                     # Add index i for divisor g.
                     # Check g <= M is implicit since g <= num <= M
                     indices_by_g[g].append(i)
                     # Add index i for the other divisor num // g if it's different from g.
                     if g*g != num:
                         divisor2 = num // g
                         # Check divisor2 <= M is implicit since divisor2 <= num <= M
                         indices_by_g[divisor2].append(i)

        total_pairs = 0
        
        # We use math.gcd directly, assuming it's efficient enough.
        # A GCD cache could be added if performance becomes an issue.

        # Iterate through all possible GCD values `g` from 1 to M.
        for g in range(1, M + 1):
            
            # Retrieve the list of indices `S_g` where element values are divisible by `g`.
            # Use .get() with default empty list to handle `g` values that might not have occurred as divisors.
            current_indices = indices_by_g.get(g, [])
            
            # If S_g is empty, no subsequences can be formed using elements divisible by `g`.
            # Thus, no pairs can have GCD equal to `g`. Skip to the next `g`.
            if not current_indices:
                 continue

            # Initialize the Dynamic Programming state.
            # dp state is a dictionary mapping (g1, g2) -> count.
            # (g1, g2) represents the GCDs of elements assigned to seq1 and seq2 respectively.
            # g1=0 or g2=0 indicates the corresponding sequence is currently empty.
            # `dp` stores the counts of ways to partition the indices processed *so far* 
            # into three sets (I1 for seq1, I2 for seq2, R for unused).
            dp = defaultdict(int)
            dp[(0, 0)] = 1 # Base case: Before processing any index, the only partition is empty sets, possible in 1 way.
            
            # Iterate through the indices in S_g. The order doesn't matter for the final counts.
            for idx in current_indices: 
                v = nums[idx] # The value at the current index.
                
                # Use a new dictionary `next_dp` to store the state after processing index `idx`.
                # This avoids issues with modifying the dictionary while iterating over it.
                next_dp = defaultdict(int)

                # Iterate through the states available *before* processing index `idx`.
                # `dp.items()` provides the states and counts from the previous step.
                for state, count in dp.items(): 
                    g1, g2 = state
                    
                    # For each state from the previous step, consider the three choices for the current index `idx`:
                    
                    # Case 1: Index `idx` is not used in either seq1 or seq2 (it goes to the 'R' set).
                    # The state (g1, g2) persists. Add its count to the same state in `next_dp`.
                    next_dp[state] = (next_dp[state] + count) % MOD

                    # Case 2: Index `idx` is added to seq1 (set I1).
                    # Calculate the new GCD for seq1. If seq1 was empty (g1=0), the new GCD is `v`.
                    # Otherwise, it's gcd(current g1, v).
                    new_g1 = math.gcd(g1, v) if g1 != 0 else v
                    # Add the count to the corresponding state in `next_dp`.
                    next_dp[(new_g1, g2)] = (next_dp[(new_g1, g2)] + count) % MOD

                    # Case 3: Index `idx` is added to seq2 (set I2).
                    # Calculate the new GCD for seq2 similarly.
                    new_g2 = math.gcd(g2, v) if g2 != 0 else v
                    # Add the count to the corresponding state in `next_dp`.
                    next_dp[(g1, new_g2)] = (next_dp[(g1, new_g2)] + count) % MOD

                # Update `dp` to the newly computed state for the next iteration.
                dp = next_dp 

            # After processing all indices relevant for the current GCD `g`,
            # the final count for pairs with GCD equal to `g` is stored in `dp[(g, g)]`.
            # The state (g, g) requires both subsequences to be non-empty (since g >= 1)
            # and have their GCDs exactly equal to `g`.
            count_g = dp.get((g, g), 0) # Use .get() for safety, although defaultdict handles missing keys.
            
            # Add the count for this `g` to the total count, modulo MOD.
            total_pairs = (total_pairs + count_g) % MOD
            
        # Return the final total count of pairs.
        return total_pairs