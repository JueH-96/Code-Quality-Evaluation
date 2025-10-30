class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        fact = [1]
        invFact = [1]
        inv = [0, 1]
        
        for i in range(1, n + 1):
            fact.append((fact[-1] * i) % MOD)
            inv.append((-(MOD // i) * inv[MOD % i]) % MOD)
            invFact.append((invFact[-1] * inv[-1]) % MOD)
        
        def C(n, k):
            if k < 0 or k > n:
                return 0
            return fact[n] * invFact[k] % MOD * invFact[n - k] % MOD
        
        def power(x, y):
            ans = 1
            while y > 0:
                if y % 2 == 1:
                    ans = (ans * x) % MOD
                x = (x * x) % MOD
                y //= 2
            return ans
        
        ans = power(26, n)
        for i in range(1, 5):
            ans = (ans - 25 * power(26, n - i) % MOD * C(n, i) % MOD + MOD) % MOD
        return ans