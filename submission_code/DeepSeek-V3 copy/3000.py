from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        sorted_list = SortedList()
        min_diff = float('inf')
        
        for i in range(len(nums)):
            if i >= x:
                sorted_list.add(nums[i - x])
            if sorted_list:
                pos = sorted_list.bisect_left(nums[i])
                if pos < len(sorted_list):
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos]))
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos - 1]))
        
        return min_diff