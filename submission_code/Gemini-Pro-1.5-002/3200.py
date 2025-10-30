class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        fact = [1] * (n + 1)
        inv = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
            inv[i] = pow(fact[i], MOD - 2, MOD)

        def nCr(n, r):
            if r < 0 or r > n:
                return 0
            num = fact[n]
            den = (inv[r] * inv[n - r]) % MOD
            return (num * den) % MOD

        ans = 0
        for l_count in range(1, n + 1):
            for e_count in range(2, n - l_count + 1):
                for t_count in range(1, n - l_count - e_count + 1):
                    remaining = n - l_count - e_count - t_count
                    ways = (l_count * e_count * (e_count - 1) // 2 * t_count) % MOD
                    ways = (ways * nCr(n, l_count)) % MOD
                    ways = (ways * nCr(n - l_count, e_count)) % MOD
                    ways = (ways * nCr(n - l_count - e_count, t_count)) % MOD
                    ways = (ways * pow(26, remaining, MOD)) % MOD
                    ways = (ways * nCr(n - l_count - e_count - t_count, remaining)) % MOD
                    ans = (ans + ways) % MOD
        
        return ans