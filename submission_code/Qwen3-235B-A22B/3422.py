class Solution:
    MOD = 10**9 + 7
    max_n = 2000  # Since n and k can be up to 1000, n + k - 1 can be up to 1999
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

    def valueAfterKSeconds(self, n: int, k: int) -> int:
        a = n + k - 1
        b = n - 1
        if b < 0 or b > a:
            return 0
        return (self.fact[a] * self.inv_fact[b] % self.MOD) * self.inv_fact[a - b] % self.MOD