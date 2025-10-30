class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_product = 0
        for x in range(2**n):
            product = ((a ^ x) * (b ^ x)) % MOD
            max_product = max(max_product, product)
        return max_product