import sys
# Setting a higher recursion depth might be necessary for deep recursive calls in compute_steps.
# The default is often 1000, which might be sufficient given popcount reduces values quickly.
# It's safer to increase it, uncomment the line below if needed.
# sys.setrecursionlimit(2000) 

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        """
        Counts the number of positive integers x < n that are k-reducible.
        n is given by its binary string s.
        An integer x is k-reducible if applying the operation "replace x with popcount(x)"
        at most k times reduces it to 1.

        Args:
            s: Binary string representation of n.
            k: Maximum number of operations allowed.

        Returns:
            The count of k-reducible numbers less than n, modulo 10^9 + 7.
        """

        MOD = 10**9 + 7
        L = len(s)

        # Memoization dictionary for storing computed steps(v) values.
        memo_real_steps = {}

        # Helper function to compute popcount (needed for older Python versions)
        def get_popcount(v):
             """Computes the population count (number of set bits) of v."""
             try:
                 # Use built-in method if available (Python 3.10+)
                 return v.bit_count()
             except AttributeError:
                 # Fallback for older Python versions
                 return bin(v).count('1')

        # Function to compute steps(v): minimum number of operations i >= 1 such that f^i(v) = 1,
        # where f(v) = popcount(v).
        def compute_steps(v):
            """Computes the number of steps to reduce v to 1 using the popcount operation."""
            # Base case: v=0 is not relevant since we consider positive integers x,
            # and popcount(x) >= 1 for x >= 1.
            if v == 1: 
                # f(1) = 1. Applying the operation once results in 1. So steps(1) = 1.
                return 1 
            
            # Return memoized result if available.
            if v in memo_real_steps: 
                return memo_real_steps[v]
            
            # Compute popcount of v.
            pc = get_popcount(v)

            # If popcount is 1, the next step reaches 1. Total steps = 1.
            if pc == 1:
                res = 1 
            else:
                # If popcount > 1, it takes 1 step to get to pc, plus the steps needed from pc.
                # Recursively call compute_steps for the popcount pc.
                res = 1 + compute_steps(pc)
            
            # Store the result in the memoization table.
            memo_real_steps[v] = res
            return res

        # Precompute steps(v) for v from 1 up to L.
        # L is the maximum possible popcount for a number x < n, where n has L bits.
        # Any number x will have popcount(x) <= L. Subsequent popcounts will be smaller.
        if L > 0: 
           # This loop ensures that steps(v) is computed and memoized for all relevant small values.
           for i in range(1, L + 1):
               compute_steps(i) 

        # Determine the set of target popcounts `p` such that any number `x` with `popcount(x) = p` is k-reducible.
        target_popcounts = set()
        
        # An integer x is k-reducible if steps(x) <= k.
        
        # Case 1: popcount(x) = 1. 
        # Then x is a power of 2. steps(x) = 1.
        # Since the problem constraint is k >= 1, steps(x)=1 <= k is always true.
        # So, all numbers x with popcount(x) = 1 are k-reducible.
        target_popcounts.add(1) 
        
        # Case 2: popcount(x) = p > 1.
        # Then steps(x) = 1 + steps(p). (Since p=popcount(x), and p > 1)
        # The condition steps(x) <= k becomes 1 + steps(p) <= k, which simplifies to steps(p) <= k-1.
        # We need to find all integers p (where 1 < p <= L) such that steps(p) <= k-1.
        if k >= 1: # This check ensures k-1 >= 0. If k=0, k-1=-1, steps(p) cannot be <= -1.
             for p in range(2, L + 1):
                 # Check if steps(p) was computed and satisfies the condition.
                 # Use .get() for safer dictionary access, although the precomputation loop should guarantee presence.
                 val = memo_real_steps.get(p) 
                 if val is not None and val <= k - 1:
                     target_popcounts.add(p)

        # Precompute factorials and modular inverse factorials for nCr calculations.
        MAX_COMB_N = L # The maximum N required for C(N, R) is L.
        fact = [1] * (MAX_COMB_N + 1)
        invfact = [1] * (MAX_COMB_N + 1)
        for i in range(1, MAX_COMB_N + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Compute modular inverse using Fermat's Little Theorem (MOD = 10^9 + 7 is prime).
        # a^(MOD-2) % MOD is the modular inverse of a modulo MOD.
        invfact[MAX_COMB_N] = pow(fact[MAX_COMB_N], MOD - 2, MOD)
        # Compute inverse factorials iteratively downwards for efficiency.
        for i in range(MAX_COMB_N - 1, -1, -1): 
            invfact[i] = (invfact[i+1] * (i+1)) % MOD
           
        # Function to compute combinations C(n, r) modulo MOD efficiently.
        def nCr_mod(n, r):
            """Computes 'n choose r' modulo MOD."""
            if r < 0 or r > n:
                return 0
            # Optimization: C(n, r) = C(n, n-r). Use the smaller value for r.
            if r > n // 2: 
                 r = n - r
            # Base cases for efficiency (though formula handles them)
            if r == 0: return 1 

            # Check bounds for precomputed arrays (should be fine here)
            if n > MAX_COMB_N:
                 return 0 # Or handle error appropriately

            # Calculate nCr using precomputed factorials and inverse factorials.
            numerator = fact[n]
            denominator = (invfact[r] * invfact[n-r]) % MOD
            return (numerator * denominator) % MOD

        # Function to count numbers x < n having exactly `target_k` set bits.
        # This uses a standard digit DP-like combinatorial approach.
        def count_with_k_bits(target_k):
            """Counts numbers x < n with exactly target_k set bits."""
            if target_k < 0: return 0 # Popcount cannot be negative.
            
            res_count = 0
            ones_so_far = 0 # Count of set bits in the prefix of s matched so far.
            
            # Iterate through the bits of s from left (MSB) to right (LSB).
            for i in range(L):
                remaining_len = L - 1 - i # Number of bits remaining after position i.
                
                # If the current bit in s is '1', we have a choice.
                if s[i] == '1':
                    # Option 1: Place '0' at the current position i.
                    # Any number formed with this prefix '...0' will be strictly less than n.
                    # We need to place exactly (target_k - ones_so_far) ones in the remaining `remaining_len` bits.
                    needed_ones = target_k - ones_so_far
                    comb = nCr_mod(remaining_len, needed_ones)
                    res_count = (res_count + comb) % MOD
                    
                    # Option 2: Place '1' at the current position i.
                    # This matches the prefix of n. We increment the count of ones used so far
                    # and continue to the next position.
                    ones_so_far += 1
                
                # If the current bit in s is '0', we must place '0' to potentially form a number <= n.
                # Placing '1' would make the number >= n. Since we need x < n, we cannot place '1'.
                # So, we must place '0'. ones_so_far does not change. Proceed to the next bit.
                
            # The loop correctly calculates the sum of counts for numbers x strictly less than n.
            # If n itself has `target_k` bits, it's not included by this logic.
            return res_count

        # Sum the counts for all target popcount values `p`.
        total_result = 0
        for p in target_popcounts:
            # We count positive integers x < n. `p` represents popcount(x), so p >= 1.
            # The count_with_k_bits function handles counting numbers x < n.
            # It implicitly handles the requirement x > 0 because popcount p >= 1 means x > 0.
            current_count = count_with_k_bits(p)
            total_result = (total_result + current_count) % MOD

        return total_result