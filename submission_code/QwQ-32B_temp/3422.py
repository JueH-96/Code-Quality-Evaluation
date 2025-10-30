class Solution:
    MOD = 10**9 + 7
    max_n = 2000  # Maximum possible value of (n-1 + k) when n and k are up to 1000
    fact = []
    inv_fact = []

    @classmethod
    def precompute(cls):
        if cls.fact:
            return
        # Precompute factorials
        cls.fact = [1] * (cls.max_n + 1)
        for i in range(1, cls.max_n + 1):
            cls.fact[i] = cls.fact[i-1] * i % cls.MOD
        # Precompute inverse factorials
        cls.inv_fact = [1] * (cls.max_n + 1)
        cls.inv_fact[cls.max_n] = pow(cls.fact[cls.max_n], cls.MOD - 2, cls.MOD)
        for i in range(cls.max_n - 1, -1, -1):
            cls.inv_fact[i] = cls.inv_fact[i + 1] * (i + 1) % cls.MOD

    def valueAfterKSeconds(self, n: int, k: int) -> int:
        self.precompute()
        m = (n - 1) + k
        # Compute combination C(m, k)
        if m < 0 or k < 0 or m < k:
            return 0
        numerator = self.fact[m]
        denominator = self.inv_fact[k] * self.inv_fact[m - k] % self.MOD
        return numerator * denominator % self.MOD