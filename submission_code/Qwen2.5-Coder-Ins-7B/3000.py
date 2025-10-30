from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        sl = SortedList()
        ans = float('inf')
        for i in range(x, len(nums)):
            sl.add(nums[i - x])
            j = sl.bisect_left(nums[i])
            if j < len(sl):
                ans = min(ans, sl[j] - nums[i])
            if j > 0:
                ans = min(ans, nums[i] - sl[j - 1])
        return ans