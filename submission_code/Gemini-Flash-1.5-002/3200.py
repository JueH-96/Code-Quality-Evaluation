class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        if n < 4:
            return 0

        fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = (fact[i - 1] * i) % MOD

        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res

        def inv(n):
            return power(n, MOD - 2)

        def nCr(n, r):
            if r > n or r < 0:
                return 0
            num = fact[n]
            den = (fact[r] * fact[n - r]) % MOD
            return (num * inv(den)) % MOD

        total_strings = power(26, n)
        
        bad_strings = 0
        
        for i in range(n - 3):
          bad_strings = (bad_strings + nCr(n-4,i) * power(25,n-4-i)) % MOD

        return (total_strings - bad_strings + MOD) % MOD