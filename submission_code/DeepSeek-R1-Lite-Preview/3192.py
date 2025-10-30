class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        from functools import lru_cache

        MOD = 10**9 + 7

        @lru_cache(maxsize=None)
        def dfs(pos, a_val, b_val):
            if pos < 0:
                return (a_val * b_val) % MOD
            # Try setting x's bit at pos to 0
            x0 = 0
            a_next = a_val
            b_next = b_val
            if (a >> pos) & 1:
                a_next ^= (1 << pos)
            if (b >> pos) & 1:
                b_next ^= (1 << pos)
            p0 = dfs(pos - 1, a_next, b_next)
            # Try setting x's bit at pos to 1
            x1 = 1 << pos
            a_next = a_val
            b_next = b_val
            if (a >> pos) & 1:
                a_next ^= (1 << pos)
            else:
                a_next ^= (1 << pos)
            if (b >> pos) & 1:
                b_next ^= (1 << pos)
            else:
                b_next ^= (1 << pos)
            p1 = dfs(pos - 1, a_next, b_next)
            return max(p0, p1)
        
        return dfs(n - 1, a, b)