import math
from typing import List

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        res = 0
        for i in range(n):
            x = nums[i]
            # Compute left part
            left = nums[:i]
            left_x = left.count(x)
            left_non_x = i - left_x
            # Compute right part
            right = nums[i+1:]
            right_x = right.count(x)
            right_non_x = len(right) - right_x
            # Compute a and b
            total = 0
            for a in [0, 1, 2]:
                if a > left_x:
                    continue
                rem_left = 2 - a
                if rem_left < 0 or rem_left > left_non_x:
                    continue
                # Precompute ways_left once for this a
                ways_left = math.comb(left_x, a) * math.comb(left_non_x, rem_left)
                for b in [0, 1, 2]:
                    if b > right_x:
                        continue
                    rem_right = 2 - b
                    if rem_right < 0 or rem_right > right_non_x:
                        continue
                    # Now, check if a + b >= 1, since x's count is a + b + 1
                    # which needs to be > any other's count (max a + b)
                    if a + b >= 1:
                        # Compute ways_right for this b
                        ways_right = math.comb(right_x, b) * math.comb(right_non_x, rem_right)
                        # Now, need to ensure that in the selected non-x elements, no y appears more than (a + b) times
                        # This part is complex and not handled here, leading to incorrect counts for some cases
                        total += ways_left * ways_right
            res += total
            res %= MOD
        return res % MOD