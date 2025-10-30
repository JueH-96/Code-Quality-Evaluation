class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7

        def calculate_product(x):
            return ((a ^ x) * (b ^ x)) % MOD

        max_product = 0
        for x in range(1 << n):
            max_product = max(max_product, calculate_product(x))

        return max_product