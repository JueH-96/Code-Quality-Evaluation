class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        mod = 10**9 + 7
        x = 0
        for i in range(n - 1, -1, -1):
            bit = 1 << i
            if (a & bit) == (b & bit):
                x |= bit
            else:
                continue
        
        return ((a ^ x) % mod * (b ^ x) % mod) % mod