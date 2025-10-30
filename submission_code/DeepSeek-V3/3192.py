class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_xor = 0
        max_product = 0
        
        # Iterate through all possible x values from 0 to 2^n - 1
        for x in range(0, 2**n):
            xor_a = a ^ x
            xor_b = b ^ x
            product = xor_a * xor_b
            if product > max_product:
                max_product = product
                max_xor = x
        
        return max_product % MOD