import sys 
# Setting recursion depth isn't necessary for this iterative/combinatorial solution,
# but potentially useful for other problems.
# sys.setrecursionlimit(2000) 

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        """
        Calculates the number of "good" arrays according to the problem definition.
        A good array arr of size n has elements in [1, m] and exactly k indices i
        (1 <= i < n) where arr[i-1] == arr[i].

        The approach is combinatorial. We construct the array element by element.
        The first element `arr[0]` can be any of the `m` values.
        For the subsequent `n-1` elements `arr[1]` to `arr[n-1]`, we consider the adjacent pairs `(arr[i-1], arr[i])`.
        There are `N = n-1` such pairs. We need exactly `k` pairs where `arr[i-1] == arr[i]`.
        This means exactly `N - k` pairs must have `arr[i-1] != arr[i]`.

        The number of ways to choose the `k` positions out of `N` where equality holds is given by the binomial coefficient C(N, k).
        
        For each of the `k` positions `i` where `arr[i] == arr[i-1]`, there is only 1 choice for `arr[i]` (it must be the same value as `arr[i-1]`). The total contribution from these choices is `1^k = 1`.
        
        For each of the `N - k` positions `i` where `arr[i] != arr[i-1]`, there are `m - 1` choices for `arr[i]` (any value from `1` to `m` except `arr[i-1]`). The total contribution from these choices is `(m - 1)^(N - k)`.

        Combining these factors:
        Total number of good arrays = (Choice for arr[0]) * (Ways to choose k equal positions) * (Choices for k equal positions) * (Choices for N-k unequal positions)
        Total number of good arrays = `m * C(N, k) * 1^k * (m - 1)^(N - k)`
        Total number of good arrays = `m * C(N, k) * (m - 1)^(N - k)`

        We need to compute this value modulo 10^9 + 7.

        Args:
            n: The size of the array. Constraints: 1 <= n <= 10^5
            m: The maximum value for array elements. Constraints: 1 <= m <= 10^5
            k: The exact number of adjacent equal elements required. Constraints: 0 <= k <= n - 1

        Returns:
            The number of good arrays modulo 10^9 + 7.
        """
        MOD = 10**9 + 7
        
        # N represents the total number of adjacent pairs in the array, which is n-1.
        N = n - 1
        
        # Constraints state 1 <= n, so N >= 0.
        # Constraints state 0 <= k <= n - 1, so 0 <= k <= N.
        
        # We need to compute combinations C(N, k). The maximum value N requires factorials up to N.
        # Maximum N can be 10^5 - 1 = 99999.
        max_fact_idx = N 
        
        # Precompute factorials modulo MOD. The array stores fact[i] = i! mod MOD.
        # Size is max_fact_idx + 1 to cover indices from 0 to max_fact_idx.
        fact = [1] * (max_fact_idx + 1)
        for i in range(1, max_fact_idx + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        # Precompute inverse factorials modulo MOD. The array stores inv_fact[i] = (i!)^(-1) mod MOD.
        inv_fact = [1] * (max_fact_idx + 1)
        
        # Check if N=0, which implies n=1. In this case fact[0]=1. inv_fact[0] should be 1.
        # The computation below works correctly even for N=0.
        
        # Compute the modular inverse of the largest factorial needed, N!, using Fermat's Little Theorem.
        # inv(a) = a^(MOD-2) mod MOD for prime MOD. 10^9 + 7 is prime.
        # pow(base, exponent, modulus) is efficient for modular exponentiation.
        inv_fact[max_fact_idx] = pow(fact[max_fact_idx], MOD - 2, MOD)

        # Compute the rest of inverse factorials iteratively downwards.
        # The relation inv(i!) = inv((i+1)!) * (i+1) mod MOD is used.
        for i in range(max_fact_idx - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

        # Helper function to compute nCr % MOD using the precomputed values.
        # This is efficient as factorial and inverse factorial lookups are O(1).
        def nCr_mod(n_val, r_val):
            """Computes nCr modulo MOD using precomputed tables."""
            # Check bounds for r: 0 <= r <= n is required for meaningful combination value.
            # Given constraints 0 <= k <= N, this check is technically redundant for the main call,
            # but good practice for a general nCr function.
            if r_val < 0 or r_val > n_val:
                return 0 # C(n, r) is 0 if r < 0 or r > n.
            
            # Calculate nCr = n! / (r! * (n-r)!) mod MOD
            # Using modular inverses: nCr = n! * inv(r!) * inv((n-r)!) mod MOD
            term1 = fact[n_val]
            term2 = inv_fact[r_val]
            term3 = inv_fact[n_val - r_val]
            
            # Compute the product modulo MOD
            result = (term1 * term2) % MOD
            result = (result * term3) % MOD
            return result

        # Calculate C(N, k) % MOD. This is the number of ways to choose the k positions
        # out of N possible adjacent pairs where arr[i-1] == arr[i].
        combinations = nCr_mod(N, k)
        
        # Calculate (m - 1)^(N - k) % MOD. This is the number of ways to fill values
        # for the N-k positions where arr[i-1] != arr[i]. At each such position i,
        # arr[i] can be any value from [1, m] except arr[i-1], giving m-1 choices.
        # The base is (m-1), exponent is (N-k), computed modulo MOD.
        # pow correctly handles base 0: pow(0, 0, MOD)=1, pow(0, positive_exp, MOD)=0.
        # This covers the edge case m=1. If m=1, m-1=0.
        # If N-k > 0, power_term = 0. If N-k=0 (i.e., k=N), power_term = 1.
        power_term = pow(m - 1, N - k, MOD)
        
        # The total count is derived from:
        # - Choosing the first element arr[0]: m choices.
        # - Choosing which k pairs are equal: C(N, k) choices.
        # - Filling values for the N-k unequal pairs: (m-1)^(N-k) choices.
        # - Filling values for the k equal pairs: 1^k = 1 choice (each must match previous).
        # Total = m * C(N, k) * (m - 1)^(N - k)
        
        # Calculate the final result step-by-step using modulo arithmetic
        # Calculate (combinations * m) % MOD first
        result_part1 = (combinations * m) % MOD
        
        # Then multiply by power_term % MOD
        final_result = (result_part1 * power_term) % MOD
        
        return final_result