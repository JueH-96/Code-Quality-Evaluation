class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Helper function to calculate nCr % MOD
        def nCr(n, r, mod):
            num = den = 1
            for i in range(r):
                num = (num * (n - i)) % mod
                den = (den * (i + 1)) % mod
            return num * pow(den, mod - 2, mod) % mod
        
        # Calculate the value of a[n - 1] after k seconds
        # This is equivalent to calculating the binomial coefficient C(n+k-1, k)
        return nCr(n + k - 1, k, MOD)