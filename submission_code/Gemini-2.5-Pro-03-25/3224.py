import sys 
# Setting recursion depth might not be necessary due to iterative power function,
# but included as a potentially useful measure in some environments.
# sys.setrecursionlimit(2000) 

from typing import List 

# Define MOD constant for modulo arithmetic operations
MOD = 10**9 + 7

# Precompute factorials up to MAX_VAL.
# The maximum number of healthy children N is n, and n <= 10^5.
# So we need factorials up to 10^5. MAX_VAL should be at least 100001.
MAX_VAL = 10**5 + 1 
fact = [1] * MAX_VAL
# Compute factorials modulo MOD iteratively
# fact[i] will store i! % MOD
for i in range(1, MAX_VAL): 
    fact[i] = (fact[i-1] * i) % MOD

# Modular Exponentiation function (iterative implementation)
def power(base, exp):
    """ Computes (base^exp) % MOD efficiently using binary exponentiation. """
    res = 1
    base %= MOD # Ensure base is within modulo range
    while exp > 0:
        # If exponent is odd, multiply base with result
        if exp % 2 == 1:
            res = (res * base) % MOD
        # Square the base
        base = (base * base) % MOD
        # Halve the exponent
        exp //= 2
    return res

# Modular Inverse function using Fermat's Little Theorem
def inverse(n):
    """ 
    Computes the modular multiplicative inverse of n modulo MOD.
    Uses Fermat's Little Theorem: n^(MOD-2) % MOD.
    Requires MOD to be a prime number, which 10^9 + 7 is.
    Also requires n not to be divisible by MOD.
    """
    # We use this function to compute inverse of factorials fact[L].
    # Since L <= N <= n <= 10^5 and MOD = 10^9 + 7, L < MOD.
    # For a prime MOD, k! % MOD is non-zero for k < MOD.
    # So fact[L] % MOD will not be 0, and the inverse exists.
    return power(n, MOD - 2)

class Solution:
    """
    Implements the solution to count the number of possible infection sequences.
    The approach uses combinatorics based on dividing the healthy children into segments
    and considering the ways to interleave infections across segments and internal ordering choices within segments.
    """
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        """
        Calculates the total number of possible infection sequences modulo 10^9 + 7.

        An infection sequence is the sequential order of positions in which all initially
        non-infected children get infected. Infection spreads from infected children to
        adjacent healthy children, one child per second.

        Args:
            n: The total number of children, positions 0 to n-1.
            sick: A 0-indexed sorted list containing the positions of initially infected children.

        Returns:
            The total number of distinct infection sequences modulo 10^9 + 7.
        """
        
        k = len(sick) # Number of initially sick children
        N = n - k # Total number of initially healthy children that will get infected

        if N == 0:
            # Base case: If there are no healthy children initially, the infection sequence is empty.
            # There is only 1 such sequence (the empty sequence).
            return 1 

        # The total number of sequences is derived from combinatorial analysis:
        # Total sequences = (Ways to interleave infections from different segments) * (Ways to order infections within segments)
        # This translates to: (Multinomial coefficient) * (Product of powers of 2 for internal choices)
        # Formula: [ N! / (L_0! * L_1! * ... * L_k!) ] * product[ 2^(L_i - 1) for i=1..k-1 if L_i > 0 ] % MOD

        # Calculate the multinomial coefficient component: N! / (L_0! * L_1! * ... * L_k!)
        # We compute this as N! * product_of_inverses(L_j!) mod MOD
        
        # Start with N! using the precomputed factorial table
        multinomial_coeff = fact[N]
        
        # Accumulator for the total exponent of the power of 2 factor.
        # This exponent comes from the internal choices within two-sided segments.
        total_exponent_pow2 = 0

        # Process the first segment of healthy children (positions 0 to sick[0]-1)
        L0 = sick[0] # Length of the first segment
        if L0 > 0:
            # Account for this segment's length in the multinomial coefficient
            # Divide by L0! by multiplying with its modular inverse
            inv_fact_L0 = inverse(fact[L0])
            multinomial_coeff = (multinomial_coeff * inv_fact_L0) % MOD
        # The first segment is one-sided (infection source only at sick[0]). No factor of 2 contribution.

        # Process the middle segments (healthy children between adjacent sick children sick[i-1] and sick[i])
        for i in range(1, k):
            # Length of the i-th middle segment (indices from sick[i-1]+1 to sick[i]-1)
            Li = sick[i] - sick[i-1] - 1 
            if Li > 0:
                # Update multinomial coefficient by dividing by Li! (using modular inverse)
                inv_fact_Li = inverse(fact[Li])
                multinomial_coeff = (multinomial_coeff * inv_fact_Li) % MOD
                
                # This is a two-sided segment (infection can come from both sick[i-1] and sick[i]).
                # If Li >= 1, there are 2^(Li-1) relative orderings of infections within this segment.
                # We sum the exponents (Li - 1) over all such middle segments.
                total_exponent_pow2 += (Li - 1)
        
        # Process the last segment of healthy children (positions sick[k-1]+1 to n-1)
        Lk = n - sick[k-1] - 1 # Length of the last segment
        if Lk > 0:
            # Update multinomial coefficient by dividing by Lk!
            inv_fact_Lk = inverse(fact[Lk])
            multinomial_coeff = (multinomial_coeff * inv_fact_Lk) % MOD
        # The last segment is one-sided (infection source only at sick[k-1]). No factor of 2 contribution.

        # Calculate the total power of 2 factor based on the accumulated exponent
        pow2_factor = power(2, total_exponent_pow2)
        
        # The final result is the product of the multinomial coefficient and the power of 2 factor, modulo MOD
        result = (multinomial_coeff * pow2_factor) % MOD
        
        return result