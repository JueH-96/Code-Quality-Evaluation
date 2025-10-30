class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        for i in range(n-1, -1, -1):
            mask = 1 << i
            if (a & mask) and (b & mask):
                continue
            elif (a & mask):
                if ((a ^ mask) & b) > (a & (b ^ mask)):
                    a ^= mask
            elif (b & mask):
                if ((b ^ mask) & a) > (b & (a ^ mask)):
                    b ^= mask
            else:
                a |= mask
                b |= mask
        return ((a % MOD) * (b % MOD)) % MOD