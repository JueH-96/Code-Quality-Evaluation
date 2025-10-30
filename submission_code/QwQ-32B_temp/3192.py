class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        x = 0
        for i in range(n-1, -1, -1):
            bit = 1 << i
            temp_x = x | bit
            current = (a ^ x) * (b ^ x)
            new_val = (a ^ temp_x) * (b ^ temp_x)
            if new_val > current:
                x = temp_x
        res = ((a ^ x) % MOD) * ((b ^ x) % MOD) % MOD
        return res