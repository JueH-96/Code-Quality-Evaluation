class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        mod = 10**9 + 7
        x = 0
        for i in range(n-1, -1, -1):
            a_i = (a >> i) & 1
            b_i = (b >> i) & 1
            if a_i == b_i:
                x_i = a_i ^ 1  # complement of a_i
            else:
                x_i = 0
            x |= (x_i << i)
        result = ((a ^ x) * (b ^ x)) % mod
        return result