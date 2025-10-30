class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        from sortedcontainers import SortedList
        
        n = len(nums)
        min_diff = float('inf')
        sorted_list = SortedList()
        
        for i in range(x, n):
            sorted_list.add(nums[i - x])
            pos = sorted_list.bisect_left(nums[i])
            
            if pos < len(sorted_list):
                min_diff = min(min_diff, abs(nums[i] - sorted_list[pos]))
            if pos > 0:
                min_diff = min(min_diff, abs(nums[i] - sorted_list[pos - 1]))
        
        return min_diff