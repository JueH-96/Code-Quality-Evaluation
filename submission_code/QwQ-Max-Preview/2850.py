from functools import lru_cache

class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        @lru_cache(maxsize=None)
        def dp(last_char, x_remaining, y_remaining, z_remaining):
            max_count = 0
            if last_char == 'A':
                if y_remaining > 0:
                    max_count = max(max_count, 1 + dp('B', x_remaining, y_remaining - 1, z_remaining))
            elif last_char == 'B':
                if x_remaining > 0:
                    max_count = max(max_count, 1 + dp('A', x_remaining - 1, y_remaining, z_remaining))
                if z_remaining > 0:
                    max_count = max(max_count, 1 + dp('B', x_remaining, y_remaining, z_remaining - 1))
            return max_count
        
        max_total = 0
        if x > 0:
            current = 1 + dp('A', x - 1, y, z)
            max_total = max(max_total, current)
        if y > 0:
            current = 1 + dp('B', x, y - 1, z)
            max_total = max(max_total, current)
        if z > 0:
            current = 1 + dp('B', x, y, z - 1)
            max_total = max(max_total, current)
        
        return max_total * 2