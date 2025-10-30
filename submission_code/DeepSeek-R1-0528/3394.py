class Solution:
    def minEnd(self, n: int, x: int) -> int:
        k = n - 1
        if k == 0:
            return x
        needed_bits = k.bit_length()
        free_bits = []
        i = 0
        while len(free_bits) < needed_bits:
            if (x & (1 << i)) == 0:
                free_bits.append(i)
            i += 1
        y = 0
        for j in range(needed_bits):
            if (k >> j) & 1:
                y |= (1 << free_bits[j])
        return x | y