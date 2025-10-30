class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        for i in range(n-1, -1, -1):
            mask = 1 << i
            if (a & mask) == 0 and (b & mask) == 0:
                a |= mask
                b |= mask
            elif (a & mask) != 0 and (b & mask) != 0:
                continue
            elif (a & mask) != 0:
                b |= mask
            else:
                a |= mask
        return (a * b) % MOD