class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7

        # Helper function to calculate (base^exp) % mod
        def pow_mod(base, exp, mod):
            result = 1
            base = base % mod
            while exp > 0:
                if exp % 2 == 1:
                    result = (result * base) % mod
                exp = exp >> 1
                base = (base * base) % mod
            return result

        # If k is 0, each element can be any of m values
        if k == 0:
            return pow_mod(m, n, MOD)

        # If k is n-1, all elements must be the same
        if k == n - 1:
            return m

        # For other cases, we need to calculate the number of ways to place k equalities
        # among n-1 possible positions (between elements)

        # Total ways to choose k positions out of n-1
        total_ways = pow_mod(m, n - k, MOD)

        # Ways to place k equalities among n-1 positions
        ways_to_place_equalities = pow_mod(m - 1, k, MOD)

        # Total good arrays
        total_good_arrays = (total_ways * ways_to_place_equalities) % MOD

        return total_good_arrays