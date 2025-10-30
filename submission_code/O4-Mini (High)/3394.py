class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # We want an increasing sequence nums[i] = x OR y_i,
        # where y_i has no bits in common with x, and y_i are sorted.
        # It can be shown the k-th smallest y is obtained by
        # mapping the bits of k into the zero‐bit positions of x.
        
        k = n - 1
        # Highest bit index in k (if k==0, J will be -1 and loops below skip)
        J = k.bit_length() - 1
        
        # Gather the first J+1 zero‐bit positions of x into list z.
        z = []
        p = 0
        while len(z) <= J:
            if ((x >> p) & 1) == 0:
                z.append(p)
            p += 1
        
        # Build y = y_{n-1} by placing the 1‐bits of k into positions z[j].
        y = 0
        for j in range(J + 1):
            if (k >> j) & 1:
                y |= 1 << z[j]
        
        # The minimal possible last element is x + y.
        return x + y