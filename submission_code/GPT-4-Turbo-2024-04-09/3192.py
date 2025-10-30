class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_val = 0
        limit = (1 << n)  # 2^n
        
        for x in range(limit):
            ax = a ^ x
            bx = b ^ x
            product = (ax * bx) % MOD
            if product > max_val:
                max_val = product
        
        return max_val