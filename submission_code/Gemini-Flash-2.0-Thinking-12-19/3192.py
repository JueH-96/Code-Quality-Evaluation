class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        x = 0
        for i in range(n - 1, -1, -1):
            next_x = x | (1 << i)
            prod1 = (a ^ x) * (b ^ x)
            prod2 = (a ^ next_x) * (b ^ next_x)
            if prod2 >= prod1:
                x = next_x
        mod_value = 10**9 + 7
        return ((a ^ x) * (b ^ x)) % mod_value