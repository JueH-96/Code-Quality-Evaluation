MOD = 10**9 + 7

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        if k >= n:
            return 0
        if k == 0:
            if n == 1:
                return m
            else:
                return m * (m - 1) ** (n - 1) % MOD
        # Precompute factorials and inverse factorials
        max_n = n
        fact = [1] * (max_n + 1)
        inv_fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        def comb(n, k):
            if k < 0 or k > n:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
        # The number of ways to choose k positions to have arr[i-1] == arr[i]
        # The remaining positions must have arr[i-1] != arr[i]
        # The first element can be any of the m choices
        # For the positions where arr[i-1] == arr[i], the value is fixed once the previous is chosen
        # For the positions where arr[i-1] != arr[i], the value can be any of the m-1 choices
        # So the total is m * C(n-1, k) * (m-1)^(n-1 - k)
        total = m * comb(n-1, k) % MOD
        total = total * pow(m-1, n-1 - k, MOD) % MOD
        return total