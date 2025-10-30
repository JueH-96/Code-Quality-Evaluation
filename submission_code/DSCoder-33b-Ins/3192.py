class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_val = (1 << n) - 1
        max_product = 0
        for x in range(max_val + 1):
            product = (a ^ x) * (b ^ x)
            max_product = max(max_product, product)
        return max_product % MOD