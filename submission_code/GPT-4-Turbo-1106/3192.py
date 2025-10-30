class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        # If a and b have different bits at the same position, we can make them both 1
        # by XORing with 0. If they have the same bits, we can make them both 1 by XORing
        # with 1 at the highest bit where they differ and 0 elsewhere.
        # So, we want to maximize the number of 1s in the product, which is achieved by
        # setting x to have 1s in all positions up to 2^n - 1.
        x = (1 << n) - 1
        
        # Calculate the product and return the result modulo MOD.
        result = ((a ^ x) * (b ^ x)) % MOD
        return result