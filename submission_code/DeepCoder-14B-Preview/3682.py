MOD = 10**9 + 7
max_n = 10**5

# Precompute factorial and inverse factorial arrays
fact = [1] * (max_n + 1)
for i in range(1, max_n + 1):
    fact[i] = fact[i-1] * i % MOD

inv_fact = [1] * (max_n + 1)
inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
for i in range(max_n - 1, -1, -1):
    inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

def comb(a, b):
    if a < 0 or b < 0 or a < b:
        return 0
    return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        r = n - k
        if r < 1 or r > n:
            return 0
        c = comb(n - 1, r - 1)
        power = pow(m - 1, r - 1, MOD)
        ans = m * c % MOD
        ans = ans * power % MOD
        return ans