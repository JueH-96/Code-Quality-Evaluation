class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_val = 0
        limit = 2**n
        
        for x in range(limit):
            val = (a ^ x) * (b ^ x)
            if val > max_val:
                max_val = val
        
        return max_val % MOD