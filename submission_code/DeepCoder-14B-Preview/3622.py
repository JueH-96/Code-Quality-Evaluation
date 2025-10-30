import bisect

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        if not nums:
            return 0
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        
        min_val = sorted_nums[0]
        max_val = sorted_nums[-1]
        min_x = min_val - k
        max_x = max_val + k
        
        max_freq = 0
        for x in range(min_x, max_x + 1):
            lower = x - k
            upper = x + k
            left = bisect.bisect_left(sorted_nums, lower)
            right = bisect.bisect_right(sorted_nums, upper)
            number_in_range = right - left
            
            count_x = bisect.bisect_right(sorted_nums, x) - bisect.bisect_left(sorted_nums, x)
            
            non_x_in_range = number_in_range - count_x
            max_adjusted = min(non_x_in_range, numOperations)
            total = count_x + max_adjusted
            
            if total > max_freq:
                max_freq = total
        
        return max_freq