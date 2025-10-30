class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials and inverse factorials for binomial coefficients
        # up to 2*x (just to be safe; x <= 1000)
        max_val = 2 * x
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Fermat's little theorem for inverse: a^(MOD-2) mod MOD
        def modinv(a, m=MOD):
            # Fast exponentiation to find inverse
            # (since m is prime, inverse of a is a^(m-2) mod m)
            result = 1
            base = a
            e = m - 2
            while e > 0:
                if e & 1:
                    result = (result * base) % m
                base = (base * base) % m
                e >>= 1
            return result
        
        inv_fact[max_val] = modinv(fact[max_val], MOD)
        for i in reversed(range(max_val)):
            inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD
        
        def comb(n_, r_):
            if r_ < 0 or r_ > n_:
                return 0
            return (fact[n_] * inv_fact[r_] % MOD) * inv_fact[n_-r_] % MOD
        
        # Precompute i^n for i from 0..x to avoid repeated exponentiation
        # arr_pow[i] = i^n mod MOD
        arr_pow = [1] * (x + 1)
        for i in range(1, x + 1):
            # fast exponentiation to compute i^n mod
            base = i
            exp = n
            power = 1
            while exp > 0:
                if exp & 1:
                    power = (power * base) % MOD
                base = (base * base) % MOD
                exp >>= 1
            arr_pow[i] = power
        
        # Use inclusion-exclusion to count surjections:
        # number of ways to distribute n distinct performers onto k distinct
        # non-empty stages: sum_{j=0..k} [(-1)^j * C(k,j) * (k-j)^n]
        
        # Then multiply by C(x, k) for choosing which k stages are used
        # and multiply by y^k for the band scores.
        
        ans = 0
        for k in range(1, x + 1):
            # Count of onto functions from n performers to k distinct bins
            # surjection_k = sum_{j=0..k} [(-1)^j * C(k, j) * (k-j)^n] mod
            surjection_k = 0
            for j in range(k + 1):
                sign = -1 if j % 2 else 1
                ways = comb(k, j) * arr_pow[k - j] % MOD
                if sign == 1:
                    surjection_k = (surjection_k + ways) % MOD
                else:
                    surjection_k = (surjection_k - ways) % MOD
            
            # Choose which k stages are used: comb(x, k)
            # Then assign scores to these k non-empty stages: y^k
            # Combine everything
            choose_stages = comb(x, k)
            
            # Compute y^k mod quickly (fast exponent)
            base = y
            exp = k
            yk = 1
            while exp > 0:
                if exp & 1:
                    yk = (yk * base) % MOD
                base = (base * base) % MOD
                exp >>= 1
            
            ways_for_k = choose_stages * surjection_k % MOD
            ways_for_k = ways_for_k * yk % MOD
            
            ans = (ans + ways_for_k) % MOD
        
        return ans % MOD