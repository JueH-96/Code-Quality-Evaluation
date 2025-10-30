class Solution:
    mod = 10**9 + 7
    maxN = 10**5
    fact = None
    inv_fact = None

    @classmethod
    def precompute(cls):
        if cls.fact is not None:
            return
        n_max = cls.maxN
        cls.fact = [1] * (n_max + 1)
        cls.inv_fact = [1] * (n_max + 1)
        for i in range(1, n_max + 1):
            cls.fact[i] = cls.fact[i - 1] * i % cls.mod
        cls.inv_fact[n_max] = pow(cls.fact[n_max], cls.mod - 2, cls.mod)
        for i in range(n_max, 0, -1):
            cls.inv_fact[i - 1] = cls.inv_fact[i] * i % cls.mod

    def nCr(self, n, r):
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        Solution.precompute()
        t = n - k
        exp = t - 1
        term1 = m
        term2 = 1 if exp == 0 else pow(m - 1, exp, self.mod)
        term3 = self.nCr(n - 1, k)
        return term1 * term2 % self.mod * term3 % self.mod