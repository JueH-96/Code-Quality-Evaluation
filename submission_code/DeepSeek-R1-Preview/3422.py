class Solution:
    MOD = 10**9 + 7
    fact = []
    inv_fact = []
    _precomputed = False

    def __init__(self):
        if not Solution._precomputed:
            self.precompute()

    @classmethod
    def precompute(cls):
        max_n = 2000
        cls.fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            cls.fact[i] = cls.fact[i-1] * i % cls.MOD

        cls.inv_fact = [1] * (max_n + 1)
        cls.inv_fact[max_n] = pow(cls.fact[max_n], cls.MOD - 2, cls.MOD)
        for i in range(max_n - 1, -1, -1):
            cls.inv_fact[i] = cls.inv_fact[i + 1] * (i + 1) % cls.MOD

        cls._precomputed = True

    def valueAfterKSeconds(self, n: int, k: int) -> int:
        a = (n - 1) + k
        b = k
        if a < b or b < 0:
            return 0
        return (self.fact[a] * self.inv_fact[b] % self.MOD) * self.inv_fact[a - b] % self.MOD