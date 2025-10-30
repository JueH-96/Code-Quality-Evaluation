from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n        = len(nums)
        sl, sm   = SortedList(nums[:x]), float("inf")
        
        for i in range(x,n):
            sl.add(nums[i])
            sm = min( sm, sl.bisect_right(nums[i]) - sl.bisect_left(nums[i])-1 )
            sl.remove(nums[i-x])
            
        return sm