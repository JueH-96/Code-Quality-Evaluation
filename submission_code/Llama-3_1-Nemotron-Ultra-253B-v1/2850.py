class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache

        allowed = {
            ('None', 'AA'): True,
            ('None', 'BB'): True,
            ('None', 'AB'): True,
            ('AA', 'AA'): False,
            ('AA', 'AB'): False,
            ('AA', 'BB'): True,
            ('AB', 'AA'): True,
            ('AB', 'BB'): False,
            ('AB', 'AB'): True,
            ('BB', 'AA'): True,
            ('BB', 'AB'): True,
            ('BB', 'BB'): False,
        }

        @lru_cache(maxsize=None)
        def dp(x, y, z, prev):
            max_count = 0
            # Try adding "AA"
            if x > 0:
                if allowed.get((prev, 'AA'), False):
                    current = 1 + dp(x - 1, y, z, 'AA')
                    max_count = max(max_count, current)
            # Try adding "BB"
            if y > 0:
                if allowed.get((prev, 'BB'), False):
                    current = 1 + dp(x, y - 1, z, 'BB')
                    max_count = max(max_count, current)
            # Try adding "AB"
            if z > 0:
                if allowed.get((prev, 'AB'), False):
                    current = 1 + dp(x, y, z - 1, 'AB')
                    max_count = max(max_count, current)
            return max_count

        return 2 * dp(x, y, z, 'None')