MOD = 10**9 + 7

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        from math import comb
        from functools import reduce

        # Precompute factorials and inverse factorials for combinatorial calculations
        max_len = n
        fact = [1] * (max_len + 1)
        inv_fact = [1] * (max_len + 1)

        for i in range(1, max_len + 1):
            fact[i] = fact[i-1] * i % MOD

        inv_fact[max_len] = pow(fact[max_len], MOD-2, MOD)
        for i in range(max_len - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

        def comb_mod(n, k):
            if k < 0 or k > n:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

        # Identify the healthy segments between sick children
        healthy_segments = []
        prev = -1
        for s in sick:
            if s - prev - 1 > 0:
                healthy_segments.append(s - prev - 1)
            prev = s
        if n - 1 - prev > 0:
            healthy_segments.append(n - 1 - prev)

        if not healthy_segments:
            return 1

        # Calculate the total number of sequences
        total = 1
        for segment in healthy_segments:
            total = total * comb_mod(segment, segment) % MOD

        # Multiply by the number of ways to interleave the segments
        # The number of ways is the multinomial coefficient (sum of lengths)! / (product of (lengths)!)
        sum_lengths = sum(healthy_segments)
        numerator = fact[sum_lengths]
        denominator = 1
        for segment in healthy_segments:
            denominator = denominator * fact[segment] % MOD
        total = total * numerator * pow(denominator, MOD-2, MOD) % MOD

        return total