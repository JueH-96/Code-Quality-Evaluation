class Solution:
    MOD = 10**9 + 7

    def __init__(self):
        max_n = 10**5
        self.fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            self.fact[i] = self.fact[i - 1] * i % self.MOD
        self.inv_fact = [1] * (max_n + 1)
        self.inv_fact[max_n] = pow(self.fact[max_n], self.MOD - 2, self.MOD)
        for i in range(max_n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % self.MOD

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        t = n - k
        if t < 1 or t > n:
            return 0
        a = n - 1
        b = t - 1
        if b < 0 or b > a:
            return 0
        comb = self.fact[a] * self.inv_fact[b] % self.MOD
        comb = comb * self.inv_fact[a - b] % self.MOD
        term1 = m % self.MOD
        term2 = pow(m - 1, t - 1, self.MOD)
        res = comb * term1 % self.MOD
        res = res * term2 % self.MOD
        return res