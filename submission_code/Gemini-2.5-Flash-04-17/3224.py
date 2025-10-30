# Standard library imports
from typing import List

# Constant for modulo
MOD = 10**9 + 7

# Maximum possible value for K (total uninfected children) is N - 1
# Maximum N is 10^5. So MAXK = 10^5 - 1.
# We need factorials up to K. So MAXN_FOR_FACT = 10^5.
# Let's use 10^5 + 1 to be safe with 0-based vs 1-based indexing and bounds.
MAXN_FOR_FACT = 10**5 + 1
fact = [1] * MAXN_FOR_FACT

# Precompute factorials modulo MOD
fact[0] = 1
for i in range(1, MAXN_FOR_FACT):
    fact[i] = (fact[i-1] * i) % MOD

# Modular exponentiation (a^b mod M)
def power(a, b):
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

# Modular inverse using Fermat's Little Theorem (a^(M-2) mod M)
def modInverse(n):
    # We only need the inverse for positive n modulo a prime M
    return power(n, MOD - 2)

# Solution class
class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        segments = []
        m = len(sick)

        # Segment 0: Uninfected children from 0 to sick[0]-1.
        # Length sick[0]. This is a one-sided segment (adjacent only to sick[0]).
        if sick[0] > 0:
            segments.append((sick[0], 'one'))

        # Middle segments: Uninfected children between sick[i] and sick[i+1].
        # Indices from sick[i]+1 to sick[i+1]-1. Length sick[i+1] - sick[i] - 1.
        # These are two-sided segments (adjacent to both sick[i] and sick[i+1]).
        for i in range(m - 1):
            gap_len = sick[i+1] - sick[i] - 1
            if gap_len > 0:
                segments.append((gap_len, 'two'))

        # Segment m: Uninfected children from sick[-1]+1 to n-1.
        # Length n - 1 - sick[-1]. This is a one-sided segment (adjacent only to sick[-1]).
        if sick[-1] < n - 1:
            segments.append((n - 1 - sick[-1], 'one'))

        # Total number of uninfected children across all segments
        K = sum(length for length, type in segments)

        # The problem is to count the number of valid infection sequences for the K uninfected children.
        # We can view this as arranging K children.
        # Children within a one-sided segment have a fixed relative infection order.
        # Children within a two-sided segment of length L > 0 have 2^(L-1) possible relative infection orderings.

        # The number of ways to interleave children from segments of lengths l_1, l_2, ..., l_p
        # is given by the multinomial coefficient K! / (l_1! l_2! ... l_p!).
        # This accounts for assigning positions to the children from each segment type.
        # If all children within a segment had a fixed order, this would be the final answer.
        result = fact[K]

        # Calculate the denominator product: product of factorials of segment lengths
        # This cancels out the permutations *within* each segment, leaving only the ways to interleave the segments as blocks.
        denom_prod = 1
        for length, type in segments:
            denom_prod = (denom_prod * fact[length]) % MOD

        # Multiply the result by the modular inverse of the denominator product
        # This effectively performs the division K! / (l_1! l_2! ... l_p!)
        inv_denom_prod = modInverse(denom_prod)
        result = (result * inv_denom_prod) % MOD

        # For two-sided segments with length L > 0, there are 2^(L-1) valid relative orderings
        # of the children within that segment. The multinomial coefficient implicitly assumed only 1
        # valid relative ordering within each segment type. We must multiply by the number of flexible
        # relative orderings available for the two-sided segments.
        two_sided_factor_prod = 1
        for length, type in segments:
            if type == 'two':
                 # Our segments list only contains lengths > 0 for middle gaps, so length >= 1 here.
                 # If length = 1, power(2, 1-1) = power(2, 0) = 1, which is correct.
                 two_sided_factor_prod = (two_sided_factor_prod * power(2, length - 1)) % MOD

        # Multiply the result by the total factor from the two-sided segments
        result = (result * two_sided_factor_prod) % MOD

        return result