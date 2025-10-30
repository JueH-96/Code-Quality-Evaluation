class Solution:
    MOD = 10**9 + 7
    max_fact = 10**5
    _fact = None
    _inv_fact = None

    @classmethod
    def _precompute(cls):
        if cls._fact is not None:
            return
        cls._fact = [1] * (cls.max_fact + 1)
        for i in range(1, cls.max_fact + 1):
            cls._fact[i] = cls._fact[i-1] * i % cls.MOD
        cls._inv_fact = [1] * (cls.max_fact + 1)
        cls._inv_fact[cls.max_fact] = pow(cls._fact[cls.max_fact], cls.MOD - 2, cls.MOD)
        for i in range(cls.max_fact - 1, -1, -1):
            cls._inv_fact[i] = cls._inv_fact[i+1] * (i+1) % cls.MOD

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        self._precompute()
        t = n - k
        a = n - 1
        b = t - 1
        if b < 0 or b > a:
            return 0
        comb = self._fact[a] * self._inv_fact[b] % self.MOD
        comb = comb * self._inv_fact[a - b] % self.MOD
        pow_part = pow(m - 1, t - 1, self.MOD)
        res = comb * m % self.MOD
        res = res * pow_part % self.MOD
        return res