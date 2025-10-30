from functools import lru_cache

class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        @lru_cache(maxsize=None)
        def dp(x, y, z, last):
            max_len = 0
            # Try adding AA
            if x > 0:
                combined = last + 'AA'
                valid = True
                if len(combined) >= 3:
                    for i in range(len(combined) - 2):
                        triplet = combined[i:i+3]
                        if triplet == 'AAA' or triplet == 'BBB':
                            valid = False
                            break
                if valid:
                    current = 2 + dp(x - 1, y, z, 'AA')
                    max_len = max(max_len, current)
            # Try adding BB
            if y > 0:
                combined = last + 'BB'
                valid = True
                if len(combined) >= 3:
                    for i in range(len(combined) - 2):
                        triplet = combined[i:i+3]
                        if triplet == 'AAA' or triplet == 'BBB':
                            valid = False
                            break
                if valid:
                    current = 2 + dp(x, y - 1, z, 'BB')
                    max_len = max(max_len, current)
            # Try adding AB
            if z > 0:
                combined = last + 'AB'
                valid = True
                if len(combined) >= 3:
                    for i in range(len(combined) - 2):
                        triplet = combined[i:i+3]
                        if triplet == 'AAA' or triplet == 'BBB':
                            valid = False
                            break
                if valid:
                    current = 2 + dp(x, y, z - 1, 'AB')
                    max_len = max(max_len, current)
            return max_len
        return dp(x, y, z, '')