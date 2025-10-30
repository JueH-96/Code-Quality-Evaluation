class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache

        pieces = {'AA': ('A', 'A'), 'BB': ('B', 'B'), 'AB': ('A', 'B')}

        @lru_cache(None)
        def dp(x, y, z, last1, last2):
            max_length = 0
            for piece, (c1, c2) in pieces.items():
                if piece == 'AA' and x > 0:
                    if not (last1 == 'A' and last2 == 'A'):
                        new_last1, new_last2 = c1, c2
                        # Check for "AAA"
                        if last1 == 'A' and last2 == 'A' and c1 == 'A':
                            continue
                        max_length = max(max_length, 2 + dp(x-1, y, z, new_last1, new_last2))
                elif piece == 'BB' and y > 0:
                    if not (last1 == 'B' and last2 == 'B'):
                        new_last1, new_last2 = c1, c2
                        # Check for "BBB"
                        if last1 == 'B' and last2 == 'B' and c1 == 'B':
                            continue
                        max_length = max(max_length, 2 + dp(x, y-1, z, new_last1, new_last2))
                elif piece == 'AB' and z > 0:
                    new_last1, new_last2 = c1, c2
                    # Check for "AAB" or "ABB"
                    if last1 == 'A' and last2 == 'A' and c1 == 'A':
                        continue
                    if last1 == 'B' and last2 == 'B' and c1 == 'B':
                        continue
                    max_length = max(max_length, 2 + dp(x, y, z-1, new_last1, new_last2))
            return max_length

        return dp(x, y, z, '', '')