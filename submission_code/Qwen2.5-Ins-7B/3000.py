from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        sl = SortedList()
        min_diff = float('inf')
        
        for i in range(x, len(nums)):
            sl.add(nums[i - x])
            min_diff = min(min_diff, abs(nums[i] - sl.bisect_left(nums[i])))
        
        return min_diff