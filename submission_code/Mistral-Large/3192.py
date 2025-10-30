class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_product = 0
        x = 0

        for i in range(n):
            bit = 1 << i
            # Try setting the bit to 1
            x1 = x | bit
            product1 = (a ^ x1) * (b ^ x1) % MOD

            # Try setting the bit to 0
            x0 = x
            product0 = (a ^ x0) * (b ^ x0) % MOD

            # Choose the bit that maximizes the product
            if product1 > product0:
                x = x1
            else:
                x = x0

            max_product = max(max_product, (a ^ x) * (b ^ x) % MOD)

        return max_product