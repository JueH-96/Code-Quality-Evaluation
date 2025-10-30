class Solution:
    def minEnd(self, n: int, x: int) -> int:
        m = x.bit_length() - 1  # Highest bit position set in x
        # Collect lower available bits (positions < m where x has 0)
        lower_bits = []
        for i in range(m):
            if not (x & (1 << i)):
                lower_bits.append(i)
        t = len(lower_bits)
        k = n - 1
        add = 0
        
        def get_bit_position(i):
            if i < t:
                return lower_bits[i]
            else:
                return (m + 1) + (i - t)
        
        # Iterate through each bit of k (0 is the LSB)
        temp = k
        index = 0
        while temp > 0:
            if temp & 1:
                add += (1 << get_bit_position(index))
            temp >>= 1
            index += 1
        
        return x + add