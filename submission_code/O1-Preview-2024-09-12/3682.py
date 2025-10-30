class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10 ** 9 + 7
        
        if m == 1:
            if k == n -1:
                return 1
            else:
                return 0
        
        max_n = n -1  # As we need factorials up to n -1
        factorial = [1] * (max_n + 2)
        inv_factorial = [1] * (max_n + 2)
        
        # Precompute factorials up to max_n
        for i in range(1, max_n + 2):
            factorial[i] = factorial[i -1] * i % MOD
        
        # Precompute inverse factorials using Fermat's little theorem
        inv_factorial[max_n +1] = pow(factorial[max_n +1], MOD -2, MOD)
        for i in range(max_n +1, 0, -1):
            inv_factorial[i -1] = inv_factorial[i] * i % MOD
        
        def comb(n, k):
            if k < 0 or k > n:
                return 0
            return factorial[n] * inv_factorial[k] % MOD * inv_factorial[n -k] % MOD
        
        c = comb(n -1, k)
        pow_m1 = pow(m -1, n -1 - k, MOD)
        ans = m * c % MOD * pow_m1 % MOD
        return ans