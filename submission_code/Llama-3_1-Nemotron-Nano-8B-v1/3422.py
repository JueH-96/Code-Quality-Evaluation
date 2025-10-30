class Solution:
    MOD = 10**9 + 7
    max_fact = 2000  # Maximum possible value of n + k - 1 is 1000 + 1000 -1 = 1999

    # Precompute factorial and inverse factorial arrays up to max_fact
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD

    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

    def valueAfterKSeconds(self, n: int, k: int) -> int:
        a = n + k - 1
        b = n - 1
        if a < 0 or b < 0 or a < b:
            return 0
        return (fact[a] * inv_fact[b] % self.MOD) * inv_fact[a - b] % self.MOD