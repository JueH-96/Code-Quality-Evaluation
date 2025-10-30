from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 1_000_000_007

        # Helper for modular exponentiation, computes (a^b) % MOD
        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res

        m = len(sick)
        total_healthy = n - m
        
        # Based on constraints, total_healthy >= 1.
        # This check is for robustness in case constraints change.
        if total_healthy == 0:
            return 1

        # Precompute factorials and their modular inverses up to n.
        # This allows for efficient O(1) calculation of combinations.
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        inv_fact = [1] * (n + 1)
        inv_fact[n] = power(fact[n], MOD - 2)
        for i in range(n - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

        # Start with the numerator of the multinomial coefficient: total_healthy!
        # This represents the total permutations of all healthy children.
        interleaving_ways = fact[total_healthy]
        
        # Product of ways for internal infections within two-sided segments
        internal_ways_prod = 1

        # Process the first segment of healthy children (before sick[0])
        len_first = sick[0]
        if len_first > 0:
            # For the multinomial coefficient, divide by len_first!
            # (equivalent to multiplying by its modular inverse).
            interleaving_ways = (interleaving_ways * inv_fact[len_first]) % MOD

        # Process middle segments (between sick[i-1] and sick[i])
        for i in range(1, m):
            length = sick[i] - sick[i-1] - 1
            if length > 0:
                # Divide by length! for the multinomial coefficient
                interleaving_ways = (interleaving_ways * inv_fact[length]) % MOD
                # For two-sided segments, there are 2^(length-1) internal sequences
                internal_ways_prod = (internal_ways_prod * power(2, length - 1)) % MOD

        # Process the last segment (after sick[m-1])
        len_last = n - 1 - sick[m-1]
        if len_last > 0:
            # Divide by len_last!
            interleaving_ways = (interleaving_ways * inv_fact[len_last]) % MOD
            
        # The final result is the product of ways to interleave and internal ways
        result = (interleaving_ways * internal_ways_prod) % MOD
        
        return result