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

            numerator = 1
            for i in range(k):
                numerator = (numerator * (n - i)) % MOD

            denominator = 1
            for i in range(1, k + 1):
                denominator = (denominator * i) % MOD

            return (numerator * power(denominator, MOD - 2)) % MOD

        if m == 1:
            return 1 if k == n - 1 else 0
        else:
            num_unequal = n - 1 - k
            if num_unequal < 0:
                return 0

            comb = combinations(n - 1, k)
            pow_term = power(m - 1, num_unequal)

            result = (comb * m) % MOD
            result = (result * pow_term) % MOD
            return result