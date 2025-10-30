class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        mod = 10**9 + 7
        for i in range(n-1, -1, -1):
            bit = 1 << i
            a_bit = a & bit
            b_bit = b & bit
            if a_bit == b_bit:
                a |= bit
                b |= bit
            else:
                if a < b:
                    a, b = b, a
                if (a - b) > (1 << i):
                    a ^= bit
                    b ^= bit
        return (a % mod) * (b % mod) % mod