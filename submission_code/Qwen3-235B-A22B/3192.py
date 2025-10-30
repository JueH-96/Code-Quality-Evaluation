class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        x = 0
        # Process each bit from highest (n-1) down to 0
        for i in reversed(range(n)):
            bit = 1 << i
            new_x = x | bit  # Try to set this bit in x
            # Calculate the product with the new x
            prod_new = (a ^ new_x) * (b ^ new_x)
            # Calculate the product with the current x
            prod_current = (a ^ x) * (b ^ x)
            if prod_new > prod_current:
                x = new_x
        # Compute the final product and return modulo 1e9+7
        result = ((a ^ x) % MOD) * ((b ^ x) % MOD) % MOD
        return result