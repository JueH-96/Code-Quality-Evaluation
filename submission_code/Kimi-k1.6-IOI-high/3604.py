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

    @classmethod
    def comb(cls, a, b):
        if a < 0 or b < 0 or a < b:
            return 0
        return cls.fact[a] * cls.inv_fact[b] % cls.MOD * cls.inv_fact[a - b] % cls.MOD

    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = self.MOD
        max_k = min(x, n)
        ans = 0
        for k in range(1, max_k + 1):
            c_x_k = self.comb(x, k)
            S = 0
            for i in range(0, k + 1):
                c_k_i = self.comb(k, i)
                term = c_k_i * pow(k - i, n, MOD)
                if i % 2 == 1:
                    term = (-term) % MOD
                S = (S + term) % MOD
            yk = pow(y, k, MOD)
            ans = (ans + c_x_k * S % MOD * yk) % MOD
        return ans