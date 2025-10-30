class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        x = 0
        for i in range(n - 1, -1, -1):
            # Try not setting the i-th bit of x
            a1 = a ^ x
            b1 = b ^ x
            product1 = a1 * b1
            
            # Try setting the i-th bit of x
            x_with_bit = x | (1 << i)
            a2 = a ^ x_with_bit
            b2 = b ^ x_with_bit
            product2 = a2 * b2
            
            # Choose the better option
            if product2 > product1:
                x = x_with_bit
        
        a_final = a ^ x
        b_final = b ^ x
        return (a_final * b_final) % MOD