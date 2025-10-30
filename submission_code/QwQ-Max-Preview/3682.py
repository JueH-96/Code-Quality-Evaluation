class Solution:
    MOD = 10**9 + 7

    def __init__(self):
        self.max_n = 10**5
        self.fact = [1] * (self.max_n + 1)
        for i in range(1, self.max_n + 1):
            self.fact[i] = self.fact[i-1] * i % self.MOD

        self.inv_fact = [1] * (self.max_n + 1)
        self.inv_fact[self.max_n] = pow(self.fact[self.max_n], self.MOD - 2, self.MOD)
        for i in range(self.max_n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % self.MOD

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = self.MOD
        if k < 0 or k > n - 1:
            return 0
        r = n - k
        if r <= 0:
            return 0
        
        a = n - 1
        b = k
        if b > a or b < 0:
            c = 0
        else:
            c = self.fact[a] * self.inv_fact[b] % MOD
            c = c * self.inv_fact[a - b] % MOD
        
        exponent = r - 1
        if exponent < 0:
            term = 0
        else:
            term = m * pow(m - 1, exponent, MOD) % MOD
        
        ans = (c * term) % MOD
        return ans