class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 0:
            return 0
        if x == 0:
            res = 0
            num = n - 1
            for i in range(num.bit_length()):
                if (num >> i) & 1:
                    res += (1 << i)
            return res
        else:
            highest_bit = x.bit_length() - 1
            low_bits = []
            for b in range(0, highest_bit + 1):
                if (x & (1 << b)) == 0:
                    low_bits.append(b)
            k = len(low_bits)
            res = 0
            num = n - 1
            for i in range(num.bit_length()):
                if (num >> i) & 1:
                    if i < k:
                        bit = low_bits[i]
                    else:
                        bit = highest_bit + 1 + (i - k)
                    res += (1 << bit)
            return x | res