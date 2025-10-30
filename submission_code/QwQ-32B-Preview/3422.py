class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        max_fact = k + n - 1
        # Precompute factorial up to max_fact
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i - 1] * i % MOD
        # Precompute inverse factorial up to max_fact
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        # Calculate combination C(k + n - 1, n - 1)
        if n - 1 > k + n - 1:
            return 0  # Invalid case, though n-1 should be <= k+n-1
        result = fact[k + n - 1] * inv_fact[n - 1] % MOD
        result = result * inv_fact[k] % MOD
        return result