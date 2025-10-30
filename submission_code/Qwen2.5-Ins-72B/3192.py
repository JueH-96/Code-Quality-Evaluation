class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        x = 0
        for i in range(n - 1, -1, -1):
            bit = 1 << i
            if (a & bit) == (b & bit):
                x |= bit
            else:
                if (a ^ x) * (b ^ x) < (a ^ (x | bit)) * (b ^ (x | bit)):
                    x |= bit
        return ((a ^ x) * (b ^ x)) % MOD