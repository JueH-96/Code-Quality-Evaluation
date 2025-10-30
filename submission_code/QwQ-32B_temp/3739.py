class Solution:
    MOD = 10**9 + 7
    max_n = 10**5
    fact = []
    inv_fact = []

    @classmethod
    def precompute(cls):
        if cls.fact:
            return
        # Compute factorial and inverse factorial arrays up to max_n
        cls.fact = [1] * (cls.max_n + 1)
        for i in range(1, cls.max_n + 1):
            cls.fact[i] = cls.fact[i-1] * i % cls.MOD
        
        cls.inv_fact = [1] * (cls.max_n + 1)
        cls.inv_fact[cls.max_n] = pow(cls.fact[cls.max_n], cls.MOD-2, cls.MOD)
        for i in range(cls.max_n -1, -1, -1):
            cls.inv_fact[i] = cls.inv_fact[i+1] * (i+1) % cls.MOD
    
    def __init__(self):
        Solution.precompute()
    
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = Solution.MOD
        T = m * n
        K = k - 2
        a = T - 2
        b = K
        if b < 0 or b > a:
            return 0
        
        # Compute combination C(a, b)
        c = a - b
        comb = Solution.fact[a] * Solution.inv_fact[b] % MOD
        comb = comb * Solution.inv_fact[c] % MOD
        
        # Compute S
        inv6 = pow(6, MOD-2, MOD)
        
        # Compute term1 and term2 mod MOD
        m_cubed = pow(m, 3, MOD)
        term_m_part = (m_cubed - m) % MOD
        term1 = (pow(n, 2, MOD) * term_m_part) % MOD
        
        n_cubed = pow(n, 3, MOD)
        term_n_part = (n_cubed - n) % MOD
        term2 = (pow(m, 2, MOD) * term_n_part) % MOD
        
        numerator = (term1 + term2) % MOD
        S = numerator * inv6 % MOD
        
        ans = (S * comb) % MOD
        return ans