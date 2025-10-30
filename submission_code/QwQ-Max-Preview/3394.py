class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        m = n - 1
        h = x.bit_length() - 1
        zeros = []
        for i in range(h):
            if (x & (1 << i)) == 0:
                zeros.append(i)
        mask = 0
        k = 0
        while m > 0:
            if m & 1:
                if k < len(zeros):
                    pos = zeros[k]
                else:
                    pos = h + 1 + (k - len(zeros))
                mask |= 1 << pos
            m >>= 1
            k += 1
        return x | mask