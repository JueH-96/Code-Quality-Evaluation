from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        sl = SortedList()
        min_diff = float('inf')
        for i in range(len(nums)):
            if i >= x:
                sl.add(nums[i - x])
            if sl:
                pos = sl.bisect_left(nums[i])
                if pos < len(sl):
                    min_diff = min(min_diff, abs(nums[i] - sl[pos]))
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - sl[pos - 1]))
        return min_diff