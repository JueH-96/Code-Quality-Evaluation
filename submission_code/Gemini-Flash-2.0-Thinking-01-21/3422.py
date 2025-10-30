class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7

        def power(a, b, p):
            res = 1
            a %= p
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % p
                a = (a * a) % p
                b //= 2
            return res

        def inverse(n, p):
            return power(n, p - 2, p)

        def nCr_mod_p(n, r, p, fact, invFact):
            if r < 0 or r > n:
                return 0
            return (fact[n] * invFact[r] * invFact[n - r]) % p

        max_val = n + k - 1
        fact = [1] * (max_val + 1)
        invFact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        invFact[max_val] = inverse(fact[max_val], MOD)
        for i in range(max_val - 1, -1, -1):
            invFact[i] = (invFact[i + 1] * (i + 1)) % MOD

        return nCr_mod_p(n + k - 1, k, MOD, fact, invFact)