class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        x = 0
        max_product = 0
        current_a = a
        current_b = b
        
        for i in range(n-1, -1, -1):
            bit = 1 << i
            x_with_bit = x | bit
            a_with_bit = current_a ^ bit
            b_with_bit = current_b ^ bit
            product_with_bit = a_with_bit * b_with_bit
            product_without_bit = current_a * current_b
            if product_with_bit > product_without_bit:
                x = x_with_bit
                current_a = a_with_bit
                current_b = b_with_bit
                max_product = product_with_bit
            else:
                max_product = product_without_bit
        
        return max_product % MOD