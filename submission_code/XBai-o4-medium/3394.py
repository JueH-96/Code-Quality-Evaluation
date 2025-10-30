class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        m = (n - 1).bit_length()
        F = []
        i = 0
        while len(F) < m:
            if not (x & (1 << i)):
                F.append(i)
            i += 1
        k = 0
        for j in range(m):
            if (n - 1) & (1 << j):
                k |= 1 << F[j]
        return x | k