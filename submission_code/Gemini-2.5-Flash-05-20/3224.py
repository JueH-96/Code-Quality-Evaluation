import math

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7

        # Precompute factorials and inverse factorials up to n (max total uninfected children is n-1)
        MAX_VAL = n + 1 
        factorial = [1] * MAX_VAL
        inv_factorial = [1] * MAX_VAL

        for i in range(2, MAX_VAL):
            factorial[i] = (factorial[i-1] * i) % MOD
        
        # Compute modular inverse of factorial[MAX_VAL - 1] using Fermat's Little Theorem
        # a^(p-2) % p is modular inverse of a mod p, for prime p
        inv_factorial[MAX_VAL - 1] = pow(factorial[MAX_VAL - 1], MOD - 2, MOD)
        
        # Compute other inverse factorials by working backwards
        # inv_factorial[i-1] = inv_factorial[i] * i % MOD
        for i in range(MAX_VAL - 2, -1, -1):
            inv_factorial[i] = (inv_factorial[i+1] * (i+1)) % MOD

        # Add sentinels to the sick array to easily identify all segments of uninfected children.
        # -1 represents a conceptual 'infected' child before position 0.
        # n represents a conceptual 'infected' child after position n-1.
        sick_ext = [-1] + sick + [n]

        total_uninfected_children = 0
        segment_lengths = []
        internal_spread_ways_product = 1

        for i in range(len(sick_ext) - 1):
            left_infected = sick_ext[i]
            right_infected = sick_ext[i+1]

            # Calculate the number of uninfected children in the current gap
            length_of_gap = right_infected - left_infected - 1

            if length_of_gap > 0:
                segment_lengths.append(length_of_gap)
                total_uninfected_children += length_of_gap
                
                # Check if this segment is an 'internal' one (bounded by two actual sick children)
                # An internal segment allows infection from both sides, leading to 2^(k-1) ways.
                # End segments (starting at 0 or ending at n-1) only allow infection from one side (1 way).
                if left_infected != -1 and right_infected != n:
                    # For an internal segment of length k, there are 2^(k-1) ways
                    internal_spread_ways_product = (internal_spread_ways_product * pow(2, length_of_gap - 1, MOD)) % MOD
                # For end segments, the ways are 1, so no multiplication is needed as product starts at 1.

        # If there are no uninfected children, only one (empty) sequence is possible.
        if total_uninfected_children == 0:
            return 1 

        # Calculate the multinomial coefficient: total_uninfected_children! / (L1! * L2! * ... * Lm!)
        # This accounts for the ways to interleave the infection of children from different segments.
        multinomial_coeff = factorial[total_uninfected_children]
        for L in segment_lengths:
            multinomial_coeff = (multinomial_coeff * inv_factorial[L]) % MOD

        # The total number of sequences is the product of the multinomial coefficient
        # and the product of ways to infect children within internal segments.
        ans = (multinomial_coeff * internal_spread_ways_product) % MOD

        return ans