class Solution:
    def minEnd(self, n: int, x: int) -> int:
        i = n - 1
        if i == 0:
            return x
        m = i.bit_length()
        mask_bits = []
        j = 0
        while len(mask_bits) < m:
            if (x & (1 << j)) == 0:
                mask_bits.append(j)
            j += 1
        a_i = 0
        for k in range(m):
            if (i >> k) & 1:
                a_i += 1 << mask_bits[k]
        return x + a_i