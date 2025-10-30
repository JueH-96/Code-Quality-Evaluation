class Solution:
    def min_end(self, n: int, x: int) -> int:
        current = x
        for _ in range(n - 1):
            mask = x
            first_zero = None
            for i in range(32):
                if (mask & (1 << i)) == 0:
                    first_zero = i
                    break
            if first_zero is None:
                next_m = (mask << 1) + (mask << 1 - 1)
            else:
                higher_bits = []
                for i in range(first_zero + 1, 32):
                    if (mask & (1 << i)) == 0:
                        higher_bits.append(i)
                if higher_bits:
                    next_higher = higher_bits[0]
                    next_m = mask | ((1 << next_higher) - 1)
                else:
                    next_m = mask | ((1 << first_zero) - 1)
            current = next_m
        return current