class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(a, b, c, last):
            if a == 0 and b == 0 and c == 0:
                return 0
            max_len = 0
            
            # Try using "AA" string
            if a > 0:
                if last == 0 or last != 1:  # start with 'A'
                    new_a = a - 1
                    new_last = 1  # ends with 'A'
                    max_len = max(max_len, 1 + dp(new_a, b, c, new_last))
            
            # Try using "BB" string
            if b > 0:
                if last == 0 or last != 2:  # start with 'B'
                    new_b = b - 1
                    new_last = 2  # ends with 'B'
                    max_len = max(max_len, 1 + dp(a, new_b, c, new_last))
            
            # Try using "AB" string
            if c > 0:
                if last == 0 or last != 1:  # start with 'A'
                    new_c = c - 1
                    new_last = 2  # ends with 'B'
                    max_len = max(max_len, 1 + dp(a, b, new_c, new_last))
            
            return max_len
        
        result = dp(x, y, z, 0)
        return result * 2