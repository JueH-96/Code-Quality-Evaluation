class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)

        # Check if t is a rotation of s
        if s + s != t + t:
            return 0

        # Find the rotation index
        for i in range(n):
            if s[i:] + s[:i] == t:
                rotation_index = i
                break

        # Calculate the number of ways to perform k operations
        # We need to find the number of ways to split the operations into two groups:
        # 1. Operations that do not change the string
        # 2. Operations that perform the required rotation
        def comb(n, k, mod):
            if k > n:
                return 0
            num = den = 1
            for i in range(k):
                num = num * (n - i) % mod
                den = den * (i + 1) % mod
            return num * pow(den, mod - 2, mod) % mod

        # The number of ways to perform k operations is the sum of combinations
        # of choosing i operations to perform the rotation and k-i operations
        # that do not change the string.
        result = 0
        for i in range(1, k + 1):
            result = (result + comb(k - 1, i - 1, MOD)) % MOD

        return result