class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        mod = 10**9 +7
        c1 = (target - 1) // 2
        if n <= c1:
            s = n * (n + 1) // 2
            s %= mod
        else:
            inv2 = pow(2, mod - 2, mod)
            s1 = c1 % mod * (c1 + 1) % mod * inv2 % mod
            n_remaining = n - c1
            # Compute s2 carefully to handle large numbers
            first_term = target % mod
            last_term = (target + n_remaining - 1) % mod
            total_terms = n_remaining % mod
            s2 = (first_term + last_term) % mod * total_terms % mod * inv2 % mod
            s = (s1 + s2) % mod
        return s