class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
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

            numerator = 1
            for i in range(k):
                numerator = (numerator * (n - i)) % MOD

            denominator = 1
            for i in range(1, k + 1):
                denominator = (denominator * i) % MOD

            return (numerator * inverse(denominator)) % MOD

        return combinations(n + k - 1, k)