class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        fact = [1] * (n + k)
        inv_fact = [1] * (n + k)

        for i in range(1, n + k):
            fact[i] = (fact[i - 1] * i) % MOD

        inv_fact[n + k - 1] = pow(fact[n + k - 1], MOD - 2, MOD)
        for i in range(n + k - 2, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

        def nCr_mod(n, r, fact, inv_fact, MOD):
            if r < 0 or r > n:
                return 0
            num = fact[n]
            den = (inv_fact[r] * inv_fact[n - r]) % MOD
            return (num * den) % MOD

        ans = nCr_mod(k + n - 1, n - 1, fact, inv_fact, MOD)
        return ans