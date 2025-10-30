import bisect

class Solution:
    def minAbsoluteDifference(self, nums, x):
        if x == 0:
            nums_sorted = sorted(nums)
            min_diff = float('inf')
            for i in range(1, len(nums_sorted)):
                diff = nums_sorted[i] - nums_sorted[i-1]
                if diff < min_diff:
                    min_diff = diff
            return min_diff
        
        sorted_list = []
        min_diff = float('inf')
        
        for j in range(len(nums)):
            if j >= x:
                val = nums[j - x]
                insert_pos = bisect.bisect_left(sorted_list, val)
                sorted_list.insert(insert_pos, val)
            
            current_val = nums[j]
            pos = bisect.bisect_left(sorted_list, current_val)
            
            if pos < len(sorted_list):
                min_diff = min(min_diff, abs(current_val - sorted_list[pos]))
            if pos > 0:
                min_diff = min(min_diff, abs(current_val - sorted_list[pos - 1]))
        
        return min_diff