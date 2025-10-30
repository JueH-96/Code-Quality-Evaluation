class Solution:
    MOD = 10**9 + 7
    max_fact = 1000
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = self.MOD
        max_m = min(x, n)
        total = 0
        for m in range(1, max_m + 1):
            # Compute C(x, m)
            c_x_m = self.fact[x] * self.inv_fact[m] % MOD
            c_x_m = c_x_m * self.inv_fact[x - m] % MOD
            
            # Compute S(m)
            s_m = 0
            for k in range(0, m + 1):
                # Compute C(m, k)
                c_m_k = self.fact[m] * self.inv_fact[k] % MOD
                c_m_k = c_m_k * self.inv_fact[m - k] % MOD
                sign = -1 if k % 2 else 1
                pow_val = pow(m - k, n, MOD)
                term = sign * c_m_k * pow_val
                s_m = (s_m + term) % MOD
            s_m = s_m % MOD
            
            # Compute y^m and accumulate
            y_pow = pow(y, m, MOD)
            term = c_x_m * s_m % MOD
            term = term * y_pow % MOD
            total = (total + term) % MOD
        
        return total % MOD