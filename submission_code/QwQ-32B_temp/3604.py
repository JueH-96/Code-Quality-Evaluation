class Solution:
    MOD = 10**9 + 7
    max_n = 1000
    max_k = 1000

    _fact = None
    _inv_fact = None
    _S = None

    @classmethod
    def _precompute(cls):
        if cls._fact is not None:
            return

        # Precompute fact and inv_fact
        fact = [1] * (cls.max_n + 1)
        for i in range(1, cls.max_n + 1):
            fact[i] = fact[i-1] * i % cls.MOD
        cls._fact = fact

        inv_fact = [1] * (cls.max_n + 1)
        inv_fact[cls.max_n] = pow(fact[cls.max_n], cls.MOD - 2, cls.MOD)
        for i in range(cls.max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % cls.MOD
        cls._inv_fact = inv_fact

        # Precompute Stirling numbers S(n, k)
        S = [[0] * (cls.max_k + 1) for _ in range(cls.max_n + 1)]
        S[0][0] = 1
        for n in range(1, cls.max_n + 1):
            for k in range(1, n + 1):
                if k > cls.max_k:
                    continue
                S[n][k] = (S[n-1][k-1] + k * S[n-1][k]) % cls.MOD
        cls._S = S

    def __init__(self):
        Solution._precompute()

    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = self.MOD
        fact = self._fact
        inv_fact = self._inv_fact
        S = self._S

        res = 0
        min_k = min(n, x)
        for k in range(1, min_k + 1):
            # Compute combination C(x, k)
            if x < k:
                comb = 0
            else:
                comb = (fact[x] * inv_fact[k] % MOD) * inv_fact[x - k] % MOD

            # Get Stirling number S(n, k)
            s_nk = S[n][k]

            # Compute term: C(x,k) * S(n,k) * k! * y^k
            term = comb * s_nk % MOD
            term = term * fact[k] % MOD  # multiply by k!
            term = term * pow(y, k, MOD) % MOD

            res = (res + term) % MOD

        return res