from typing import List

class Solution:
    MOD = 10**9 + 7
    _max_fact = 0
    fact = []
    inv_fact = []

    @classmethod
    def _precompute(cls, n_max: int) -> None:
        if cls._max_fact >= n_max:
            return
        # Compute factorial and inverse factorial up to n_max
        cls.fact = [1] * (n_max + 1)
        for i in range(1, n_max + 1):
            cls.fact[i] = cls.fact[i-1] * i % cls.MOD
        cls.inv_fact = [1] * (n_max + 1)
        cls.inv_fact[n_max] = pow(cls.fact[n_max], cls.MOD - 2, cls.MOD)
        for i in range(n_max - 1, -1, -1):
            cls.inv_fact[i] = cls.inv_fact[i + 1] * (i + 1) % cls.MOD
        cls._max_fact = n_max

    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        self._precompute(n)
        segments = []
        if not sick:
            return 0  # According to constraints, this won't happen

        # First segment before the first sick
        first = sick[0]
        if first > 0:
            m = first
            segments.append((m, 1))

        # Middle segments between consecutive sick
        for i in range(len(sick) - 1):
            s_prev = sick[i]
            s_next = sick[i + 1]
            distance = s_next - s_prev
            if distance > 1:
                m = distance - 1
                w = pow(2, m - 1, self.MOD)
                segments.append((m, w))

        # Last segment after the last sick
        last = sick[-1]
        if last < n - 1:
            m = (n - 1) - last
            segments.append((m, 1))

        product_W = 1
        total_m = 0
        denom_terms = 1
        for m, w in segments:
            product_W = product_W * w % self.MOD
            total_m += m
            denom_terms = denom_terms * self.inv_fact[m] % self.MOD

        if total_m == 0:
            return 1

        multinomial = self.fact[total_m] * denom_terms % self.MOD
        ans = product_W * multinomial % self.MOD
        return ans