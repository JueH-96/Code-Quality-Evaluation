from bisect import bisect_left, insort
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        ans = float('inf')
        # We'll use a sorted list (sl) to store candidate values from indices that are valid.
        sl = []
        
        # Handle the x = 0 case separately
        if x == 0:
            # For any i, valid partner indices are all j < i.
            for i in range(n):
                if sl:
                    pos = bisect_left(sl, nums[i])
                    if pos < len(sl):
                        ans = min(ans, abs(sl[pos] - nums[i]))
                    if pos > 0:
                        ans = min(ans, abs(nums[i] - sl[pos-1]))
                # Insert current value so that future indices can pair with nums[i].
                insort(sl, nums[i])
            return ans
        
        # When x > 0:
        # We want for current index i, valid partner indices are j in [0, i-x].
        # We'll insert elements into our sorted list when index i reaches j + x.
        for i in range(n):
            # When i >= x, we add nums[i - x] as a candidate for future indices.
            if i >= x:
                insort(sl, nums[i-x])
            # For index i, if there is any candidate, check for best match.
            if sl:
                pos = bisect_left(sl, nums[i])
                if pos < len(sl):
                    ans = min(ans, abs(sl[pos] - nums[i]))
                if pos > 0:
                    ans = min(ans, abs(nums[i] - sl[pos-1]))
        return ans