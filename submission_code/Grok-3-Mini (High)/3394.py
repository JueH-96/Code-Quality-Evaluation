class Solution:
    def minEnd(self, n: int, x: int) -> int:
        m = n - 1
        # Compute free bit positions where x has bit 0, using a sufficient range
        free_pos = [b for b in range(60) if (x & (1 << b)) == 0]
        # Compute z_m
        z = 0
        for i in range(27):  # m < 2^27, so check bits up to 26
            if (m >> i) & 1:
                bit_pos = free_pos[i]
                z |= (1 << bit_pos)
        # The minimum possible value of nums[n-1] is x + z
        return x + z