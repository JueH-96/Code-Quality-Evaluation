class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Find the highest set bit in x
        highest_bit = -1
        temp = x
        while temp > 0:
            highest_bit += 1
            temp >>= 1
        
        # Collect lower free bits (bits not set in x up to highest_bit)
        lower_free_bits = []
        for i in range(highest_bit + 1):
            if not (x & (1 << i)):
                lower_free_bits.append(i)
        
        m = len(lower_free_bits)
        k = n - 1
        s = 0
        j = 0
        
        while k > 0:
            if k & 1:
                if j < m:
                    s += (1 << lower_free_bits[j])
                else:
                    s += (1 << (highest_bit + 1 + (j - m)))
            k >>= 1
            j += 1
        
        return x + s