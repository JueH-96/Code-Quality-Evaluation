class Solution:
    def stringCount(self, n: int) -> int:
        mod = 10**9 + 7
        T0 = pow(26, n, mod)
        T1 = (3 * pow(25, n, mod) + n * pow(25, n - 1, mod)) % mod
        T2 = (3 * pow(24, n, mod) + 2 * n * pow(24, n - 1, mod)) % mod
        T3 = (pow(23, n, mod) + n * pow(23, n - 1, mod)) % mod
        ans = (T0 - T1 + T2 - T3) % mod
        return ans if ans >= 0 else ans + mod