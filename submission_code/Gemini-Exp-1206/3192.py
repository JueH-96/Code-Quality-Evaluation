class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        mod = 10**9 + 7
        x = 0
        for i in range(n - 1, -1, -1):
            bit_a = (a >> i) & 1
            bit_b = (b >> i) & 1
            if bit_a == bit_b:
                x |= (1 << i)
            else:
                if a > b:
                    a, b = b, a
        mask = (1 << n) - 1
        a_xor_x = a ^ x
        b_xor_x = b ^ x
        a_xor_x_flipped = a_xor_x ^ mask
        b_xor_x_flipped = b_xor_x ^ mask
        if a_xor_x_flipped > a_xor_x and b_xor_x_flipped > b_xor_x:
            return (a_xor_x_flipped % mod) * (b_xor_x_flipped % mod) % mod
        else:
            return (a_xor_x % mod) * (b_xor_x % mod) % mod