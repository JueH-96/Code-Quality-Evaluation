import math

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 1000000007
        n = len(s)
        if s not in (t + t):
            return 0
        # Get sorted divisors of n
        divs = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        divs.sort()
        # Find minimal d such that s[i] == s[(i+d) % n] for all i
        for d in divs:
            if all(s[i] == s[(i + d) % n] for i in range(n)):
                min_d = d
                break
        m = n // min_d
        # Check if s == t
        is_same = (s == t)
        # Compute powers
        pow_n_minus_1_k = pow(n - 1, k, MOD)
        pow_minus1_k = 1 if k % 2 == 0 else (MOD - 1)
        inv_n = pow(n, MOD - 2, MOD)
        # Compute num_a and a_k_mod
        num_a = (pow_n_minus_1_k - pow_minus1_k + MOD) % MOD
        a_k_mod = (num_a * inv_n) % MOD
        # Compute num_b and b_k_mod
        prod_part_b = ((n - 1) * pow_minus1_k) % MOD
        num_b = (pow_n_minus_1_k + prod_part_b) % MOD
        b_k_mod = (num_b * inv_n) % MOD
        # Compute ways
        if is_same:
            ways = (b_k_mod + ((m - 1) * a_k_mod % MOD) % MOD) % MOD
        else:
            ways = ((m * a_k_mod % MOD)) % MOD
        return ways