class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 1000000007
        for i in range(n - 1, -1, -1):
            mask = 1 << i
            if ((a & mask) == 0 and (b & mask) == 0) or ((a & mask) != 0 and (b & mask) != 0):
                continue
            if ((a ^ b) * (a ^ mask) * (b ^ mask)) > (a * b):
                if (a & mask) == 0:
                    a ^= mask
                else:
                    b ^= mask
        return (a * b) % MOD