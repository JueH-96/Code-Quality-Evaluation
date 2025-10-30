class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        x = 0
        MOD = 10**9 + 7
        for i in reversed(range(n)):
            mask = (1 << i) - 1
            # Option 0: keep x's i-th bit as 0
            x0 = x
            a0 = (a ^ x0) & (~mask)
            b0 = (b ^ x0) & (~mask)
            product0 = (a0 | mask) * (b0 | mask)
            
            # Option 1: set x's i-th bit to 1
            x1 = x | (1 << i)
            a1 = (a ^ x1) & (~mask)
            b1 = (b ^ x1) & (~mask)
            product1 = (a1 | mask) * (b1 | mask)
            
            if product1 > product0:
                x = x1
        aa = a ^ x
        bb = b ^ x
        return (aa * bb) % MOD