class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res

        def combinations(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            if k > n // 2:
                k = n - k
            
            num = 1
            den = 1
            for i in range(k):
                num = (num * (n - i)) % MOD
                den = (den * (i + 1)) % MOD
            
            return (num * power(den, MOD - 2)) % MOD

        return (combinations(n - 1, k) * m * power(m - 1, n - 1 - k)) % MOD