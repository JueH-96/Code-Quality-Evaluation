class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_product = 0
        
        # Iterate through all possible values of x from 0 to 2^n - 1
        for x in range(1 << n):
            product = (a ^ x) * (b ^ x)
            max_product = max(max_product, product)
        
        return max_product % MOD