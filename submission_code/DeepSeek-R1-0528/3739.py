mod = 10**9 + 7
max_val = 100000
fact = None
inv_fact = None

def precompute():
    global fact, inv_fact, mod, max_val
    fact = [1] * (max_val + 1)
    for i in range(1, max_val + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact = [1] * (max_val + 1)
    inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
    for i in range(max_val, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % mod

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        global fact, inv_fact
        if fact is None:
            precompute()
        N = m * n
        r = k - 2
        if r < 0 or r > N - 2:
            binom = 0
        else:
            binom = fact[N - 2] * inv_fact[r] % mod * inv_fact[N - 2 - r] % mod
        inv6 = pow(6, mod - 2, mod)
        S_x = (m - 1) * m % mod * (m + 1) % mod * inv6 % mod
        S_y = (n - 1) * n % mod * (n + 1) % mod * inv6 % mod
        Total_x = n * n % mod * S_x % mod
        Total_y = m * m % mod * S_y % mod
        total_distance = (Total_x + Total_y) % mod
        ans = total_distance * binom % mod
        return ans