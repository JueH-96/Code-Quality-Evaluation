class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        k = n - 1
        zero_positions = []
        for bit_index in range(60):
            if (x >> bit_index) & 1 == 0:
                zero_positions.append(bit_index)
        res = x
        for j in range(60):
            if j >= len(zero_positions):
                break
            if (k >> j) & 1:
                res |= (1 << zero_positions[j])
        return res