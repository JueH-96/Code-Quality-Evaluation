class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        mask = (1 << n) - 1
        a_masked = a & mask
        b_masked = b & mask
        a_unmasked = a & ~mask
        b_unmasked = b & ~mask
        
        for i in range(n-1, -1, -1):
            bit = 1 << i
            if (a_masked & bit) == (b_masked & bit):
                a_masked |= bit
                b_masked |= bit
            elif (a_unmasked * b_unmasked + a_unmasked * b_masked + a_masked * b_unmasked) < \
                 (a_unmasked * b_unmasked + (a_masked ^ bit) * (b_unmasked + b_masked) + (b_masked ^ bit) * a_unmasked):
                a_masked ^= bit
                b_masked ^= bit
        
        return ((a_unmasked + a_masked) * (b_unmasked + b_masked)) % MOD