class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = (fact[i-1] * i) % MOD
            
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
            if r < 0 or r > n:
                return 0
            num = fact[n]
            den = (fact[r] * fact[n-r]) % MOD
            return (num * inv(den)) % MOD
        
        ans = 1
        prev = -1
        
        for i in sick:
            length = i - prev - 1
            if length > 0:
                ans = (ans * nCr(length, 1)) % MOD
                ans = (ans * fact[length -1]) % MOD
            prev = i
            
        length = n - prev - 1
        if length > 0:
            ans = (ans * nCr(length, 0)) % MOD
            ans = (ans * fact[length]) % MOD
            
        return ans