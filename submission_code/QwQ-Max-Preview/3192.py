class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        mod = 10**9 + 7
        x = 0
        for i in reversed(range(n)):
            bit = 1 << i
            a_xor = a ^ x
            b_xor = b ^ x
            new_a = a_xor ^ bit
            new_b = b_xor ^ bit
            if new_a * new_b > a_xor * b_xor:
                x |= bit
        return ((a ^ x) * (b ^ x)) % mod