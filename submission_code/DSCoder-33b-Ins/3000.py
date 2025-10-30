class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        from sortedcontainers import SortedList
        sl = SortedList()
        res = float('inf')
        for i in range(x, len(nums)):
            sl.add(nums[i - x])
            idx = sl.bisect_left(nums[i])
            if idx < len(sl):
                res = min(res, sl[idx] - nums[i])
            if idx > 0:
                res = min(res, nums[i] - sl[idx - 1])
        return res