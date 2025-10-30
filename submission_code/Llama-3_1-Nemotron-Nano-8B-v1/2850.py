from functools import lru_cache

class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        @lru_cache(maxsize=None)
        def dp(a, b, c, prev_prev1, prev2):
            if a < 0 or b < 0 or c < 0:
                return 0
            max_len = 0
            # Try adding AA
            if a > 0:
                # Check if adding AA would create forbidden triplets
                triplet1 = (prev_prev1, prev2, 'A')
                triplet2 = (prev2, 'A', 'A')
                if not (self.is_forbidden(triplet1) or self.is_forbidden(triplet2)):
                    new_prev_prev1 = prev2
                    new_prev2 = 'A'
                    current_len = 2 + dp(a-1, b, c, new_prev_prev1, new_prev2)
                    if current_len > max_len:
                        max_len = current_len
            # Try adding BB
            if b > 0:
                triplet1 = (prev_prev1, prev2, 'B')
                triplet2 = (prev2, 'B', 'B')
                if not (self.is_forbidden(triplet1) or self.is_forbidden(triplet2)):
                    new_prev_prev1 = prev2
                    new_prev2 = 'B'
                    current_len = 2 + dp(a, b-1, c, new_prev_prev1, new_prev2)
                    if current_len > max_len:
                        max_len = current_len
            # Try adding AB
            if c > 0:
                triplet1 = (prev_prev1, prev2, 'A')
                triplet2 = (prev2, 'A', 'B')
                if not (self.is_forbidden(triplet1) or self.is_forbidden(triplet2)):
                    new_prev_prev1 = prev2
                    new_prev2 = 'B'
                    current_len = 2 + dp(a, b, c-1, new_prev_prev1, new_prev2)
                    if current_len > max_len:
                        max_len = current_len
            return max_len

        def is_forbidden(triplet):
            return triplet == ('A', 'A', 'A') or triplet == ('B', 'B', 'B')

        return dp(x, y, z, None, None)