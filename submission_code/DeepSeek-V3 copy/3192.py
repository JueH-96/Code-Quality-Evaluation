class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_xor = (1 << n) - 1
        max_product = 0
        
        # Iterate over all possible x values
        for x in range(max_xor + 1):
            xor_a = a ^ x
            xor_b = b ^ x
            product = (xor_a * xor_b) % MOD
            if product > max_product:
                max_product = product
        
        return max_product