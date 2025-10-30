class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        mod = 10**9 + 7
        max_val = 0
        for x in range(2**n):
            max_val = max(max_val, (a ^ x) * (b ^ x))
        return max_val % mod