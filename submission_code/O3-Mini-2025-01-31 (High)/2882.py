class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = 10**9 + 7
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dfs(remaining, current):
            # If we've exactly reached 0, that's one valid way
            if remaining == 0:
                return 1
            # If remaining becomes negative, this path isn't valid
            if remaining < 0:
                return 0
            count = 0
            # Try using each number starting from 'current'
            # Stop when current**x > remaining because any further numbers will also exceed the remaining sum.
            i = current
            while i ** x <= remaining:
                count = (count + dfs(remaining - i ** x, i + 1)) % mod
                i += 1
            return count

        return dfs(n, 1)