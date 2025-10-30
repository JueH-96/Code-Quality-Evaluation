class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(last_char, x_remain, y_remain, z_remain):
            if x_remain == 0 and y_remain == 0 and z_remain == 0:
                return 0
            max_count = 0

            # Try adding "AA"
            if x_remain > 0 and last_char != 'A':
                new_last = 'A'
                current = 1 + dp(new_last, x_remain - 1, y_remain, z_remain)
                if current > max_count:
                    max_count = current

            # Try adding "BB"
            if y_remain > 0 and last_char != 'B':
                new_last = 'B'
                current = 1 + dp(new_last, x_remain, y_remain - 1, z_remain)
                if current > max_count:
                    max_count = current

            # Try adding "AB"
            if z_remain > 0 and last_char != 'A':
                new_last = 'B'
                current = 1 + dp(new_last, x_remain, y_remain, z_remain - 1)
                if current > max_count:
                    max_count = current

            return max_count

        max_total = 0

        # Consider all possible starting strings
        if x > 0:
            current = 1 + dp('A', x - 1, y, z)
            if current > max_total:
                max_total = current
        if y > 0:
            current = 1 + dp('B', x, y - 1, z)
            if current > max_total:
                max_total = current
        if z > 0:
            current = 1 + dp('B', x, y, z - 1)
            if current > max_total:
                max_total = current

        return max_total * 2