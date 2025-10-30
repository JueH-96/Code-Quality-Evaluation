class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        max_x = (1 << n) - 1
        max_product = 0

        # Since n can be up to 50, iterating over all possible x is not feasible.
        # Instead, we can try to find x that maximizes (a ^ x) * (b ^ x)
        # This can be approached by setting each bit of x from high to low.

        x = 0
        for i in reversed(range(n)):
            # Try setting the i-th bit to 1
            candidate = x | (1 << i)
            if candidate > max_x:
                continue
            c = a ^ candidate
            d = b ^ candidate
            current_product = c * d
            # If setting the bit to 1 gives a better or equal product, keep it
            if current_product > max_product:
                x = candidate
                max_product = current_product
        return max_product % MOD