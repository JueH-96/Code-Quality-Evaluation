from sortedcontainers import SortedList

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        sl = SortedList()
        res = i = 0
        for j, n in enumerate(nums):
            i = max(i, sl.bisect_left(n-2))
            res += j - i + 1
            sl.add(n)
        return res