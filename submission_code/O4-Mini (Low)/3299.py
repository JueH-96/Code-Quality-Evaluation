from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        cnt = Counter(nums)
        ans = 1
        
        # Special case for base = 1
        c1 = cnt.get(1, 0)
        if c1 > 0:
            # we can use 2*m+1 <= c1, so max odd length <= c1
            if c1 % 2 == 1:
                ans = max(ans, c1)
            else:
                ans = max(ans, c1 - 1)
        
        max_val = max(nums)
        # for each possible base x > 1
        for x in cnt:
            if x <= 1:
                continue
            v = x
            levels = []
            # build the chain until missing or overflow
            while v <= max_val:
                c = cnt.get(v, 0)
                if c == 0:
                    break
                levels.append(c)
                # if count < 2, we can only use this as the top
                if c < 2:
                    break
                # next power: square
                # avoid infinite loop, though for x > 1 v will grow
                v = v * v
                if v > 10**9:
                    break
            
            # if we found at least one level, compute length = 2*m+1 where
            # m = number_of_levels - 1
            if levels:
                m = len(levels) - 1
                length = 2*m + 1
                ans = max(ans, length)
        
        return ans