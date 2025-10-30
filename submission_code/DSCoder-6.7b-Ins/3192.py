class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_xor = 0
        for x in range(2**n):
            xor = (a ^ x) * (b ^ x)
            if xor > max_xor:
                max_xor = xor
        return max_xor % MOD