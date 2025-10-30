class Solution:
    def stringCount(self, n: int) -> int:
        mod = 10**9 + 7
        term1 = pow(26, n, mod)
        term2 = (3 * pow(25, n, mod)) % mod
        term3 = (n * pow(25, n - 1, mod)) % mod if n >= 1 else 0
        term4 = (3 * pow(24, n, mod)) % mod
        term5 = (2 * n * pow(24, n - 1, mod)) % mod if n >= 1 else 0
        term6 = pow(23, n, mod)
        term7 = (n * pow(23, n - 1, mod)) % mod if n >= 1 else 0
        total_good = (term1 - term2 - term3 + term4 + term5 - term6 - term7 + 3 * mod) % mod
        return total_good