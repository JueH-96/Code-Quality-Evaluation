class Solution:
    def waysToReachStair(self, k: int) -> int:
        # Compute C(n, r)
        def comb(n: int, r: int) -> int:
            if r < 0 or r > n:
                return 0
            # use symmetry
            if r > n - r:
                r = n - r
            c = 1
            for i in range(1, r + 1):
                c = c * (n - r + i) // i
            return c

        res = 0
        m = 0
        # We seek all m such that 2^m >= k and 2^m - k <= m+1.
        # Once 2^m > k + (m+1), no larger m can satisfy the second condition.
        while True:
            two_pow_m = 1 << m
            if two_pow_m > k + m + 1:
                break
            if two_pow_m >= k:
                downs = two_pow_m - k
                # downs = number of down moves n must fit into m+1 slots
                if downs <= m + 1:
                    res += comb(m + 1, downs)
            m += 1

        return res