class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_value = 0
        limit = 1 << n  # This is 2^n
        
        # Iterate over all possible values of x from 0 to 2^n - 1
        for x in range(limit):
            # Calculate (a XOR x) and (b XOR x)
            ax = a ^ x
            bx = b ^ x
            # Calculate the product and update max_value if it's larger
            product = ax * bx
            if product > max_value:
                max_value = product
        
        # Return the maximum value modulo 10^9 + 7
        return max_value % MOD