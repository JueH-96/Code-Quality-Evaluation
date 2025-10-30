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

        def inv(n):
            return power(n, MOD - 2)

        dp = {}
        def combinations(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            if (n, k) in dp:
                return dp[(n,k)]
            res = (combinations(n - 1, k - 1) + combinations(n - 1, k)) % MOD
            dp[(n,k)] = res
            return res

        ans = 0
        for i in range(m):
            ans = (ans + combinations(n - 1, k) * power(m -1, n - 1 -k) ) % MOD

        return (ans * m) % MOD