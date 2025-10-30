class Solution:
    def stringCount(self, n: int) -> int:
        mod = 10**9 + 7
        total = pow(26, n, mod)
        t25 = pow(25, n, mod)
        t25_1 = pow(25, n-1, mod)
        t24 = pow(24, n, mod)
        t24_1 = pow(24, n-1, mod)
        t23 = pow(23, n, mod)
        t23_1 = pow(23, n-1, mod)
        
        bad = (3 * t25 + n * t25_1 - 3 * t24 - 2 * n * t24_1 + t23 + n * t23_1) % mod
        ans = (total - bad) % mod
        return ans