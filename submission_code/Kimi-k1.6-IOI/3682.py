class Solution:
    def __init__(self):
        self.MOD = 10**9 + 7
        max_n = 10**5
        self.fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            self.fact[i] = self.fact[i-1] * i % self.MOD
        
        self.inv_fact = [1] * (max_n + 1)
        self.inv_fact[max_n] = pow(self.fact[max_n], self.MOD - 2, self.MOD)
        for i in range(max_n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i+1] * (i+1) % self.MOD
    
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = self.MOD
        if k < 0 or k > n - 1:
            return 0
        
        # Calculate C(n-1, k)
        C = self.fact[n-1] * self.inv_fact[k] % MOD
        C = C * self.inv_fact[n-1 - k] % MOD
        
        exponent = n - k - 1
        power = pow(m - 1, exponent, MOD) if exponent >= 0 else 0
        
        ans = C * m % MOD
        ans = ans * power % MOD
        
        return ans