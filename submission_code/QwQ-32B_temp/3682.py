class Solution:
    MOD = 10**9 + 7
    max_n = 10**5
    fact = []
    inv_fact = []

    @classmethod
    def precompute(cls):
        if cls.fact:
            return
        # Initialize fact array
        cls.fact = [1] * (cls.max_n + 1)
        for i in range(1, cls.max_n + 1):
            cls.fact[i] = cls.fact[i-1] * i % cls.MOD
        
        # Initialize inv_fact array
        cls.inv_fact = [1] * (cls.max_n + 1)
        cls.inv_fact[cls.max_n] = pow(cls.fact[cls.max_n], cls.MOD - 2, cls.MOD)
        for i in range(cls.max_n - 1, -1, -1):
            cls.inv_fact[i] = cls.inv_fact[i + 1] * (i + 1) % cls.MOD

    def __init__(self):
        Solution.precompute()

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        R = n - k
        if R < 1 or R > n:
            return 0
        
        a = n - 1
        b = R - 1
        if b < 0 or b > a:
            return 0
        
        # Compute combination C(a, b)
        comb = Solution.fact[a] * Solution.inv_fact[b] % MOD
        comb = comb * Solution.inv_fact[a - b] % MOD
        
        # Compute m * (m-1)^(R-1)
        power_term = pow(m - 1, R - 1, MOD)
        term = m * power_term % MOD
        
        ans = comb * term % MOD
        return ans