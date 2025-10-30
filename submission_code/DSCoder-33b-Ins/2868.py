from typing import List
from sortedcontainers import SortedList

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        sl = SortedList()
        left = 0
        res = 0
        for right in range(len(nums)):
            sl.add(nums[right])
            while sl[-1] - sl[0] > 2:
                sl.remove(nums[left])
                left += 1
            res += right - left + 1
        return res