from sortedcontainers import SortedList
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            return 0
        
        sl = SortedList()
        min_diff = float('inf')
        
        for i in range(x, len(nums)):
            sl.add(nums[i - x])
            idx = sl.bisect_left(nums[i])
            
            if idx < len(sl):
                min_diff = min(min_diff, abs(nums[i] - sl[idx]))
            if idx > 0:
                min_diff = min(min_diff, abs(nums[i] - sl[idx - 1]))
                
        return min_diff