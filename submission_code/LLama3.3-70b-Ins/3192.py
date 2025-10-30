class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_product = 0
        
        # Generate all possible values of x
        for x in range(2**n):
            # Calculate (a XOR x) and (b XOR x)
            ax = a ^ x
            bx = b ^ x
            
            # Calculate the product of (a XOR x) and (b XOR x)
            product = ax * bx
            
            # Update the maximum product
            max_product = max(max_product, product)
        
        # Return the maximum product modulo 10^9 + 7
        return max_product % MOD