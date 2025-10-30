class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        x = 0
        for i in reversed(range(n)):
            candidate = x | (1 << i)
            current_val = (a ^ x) * (b ^ x)
            new_val = (a ^ candidate) * (b ^ candidate)
            if new_val >= current_val:
                x = candidate
        return ((a ^ x) * (b ^ x)) % MOD