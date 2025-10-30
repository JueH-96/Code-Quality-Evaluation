class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7

        maxN = n + k + 1
        factorial = [1] * (maxN)
        inv_factorial = [1] * (maxN)

        for i in range(1, maxN):
            factorial[i] = (factorial[i - 1] * i) % MOD

        # Compute inverse factorial using Fermat's Little Theorem
        inv_factorial[maxN - 1] = pow(factorial[maxN - 1], MOD - 2, MOD)
        for i in range(maxN - 2, -1, -1):
            inv_factorial[i] = (inv_factorial[i + 1] * (i + 1)) % MOD

        def comb(a, b):
            if b < 0 or b > a:
                return 0
            return (factorial[a] * inv_factorial[b] % MOD) * inv_factorial[a - b] % MOD

        return comb(n + k - 1, n - 1)