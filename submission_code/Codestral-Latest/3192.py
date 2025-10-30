class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7

        # Function to calculate (a XOR x) * (b XOR x) % MOD
        def calculate_product(x):
            axorx = a ^ x
            bxorx = b ^ x
            return (axorx * bxorx) % MOD

        max_product = 0
        for x in range(2**n):
            max_product = max(max_product, calculate_product(x))

        return max_product