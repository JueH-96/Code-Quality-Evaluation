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

        def inverse(n):
            return power(n, MOD - 2)

        def combinations(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            if k > n // 2:
                k = n - k

            res = 1
            for i in range(k):
                res = (res * (n - i)) % MOD
                res = (res * inverse(i + 1)) % MOD
            return res

        if k + 1 > n:
            return 0

        ans = 0
        for x in range(1, m + 1):
            term1 = combinations(n - 1, k)
            term2 = power(m - 1, n - 1 - k)
            term3 = power(x - 1, n - 1 - k)
            
            ans = (ans + (term1 * term2) % MOD) % MOD
            ans = (ans - (term1 * term3) % MOD + MOD) % MOD
            
        
        return ans