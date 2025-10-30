class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_val = 0
        for x in range(2**n):
            val = (a ^ x) * (b ^ x)
            max_val = max(max_val, val)
        return max_val % MOD