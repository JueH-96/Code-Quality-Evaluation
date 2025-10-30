class Solution:
    def minDifference(self, nums: List[int]) -> int:
        def find_min_max(nums):
            min_val = float('inf')
            max_val = float('-inf')
            for num in nums:
                if num != -1:
                    min_val = min(min_val, num)
                    max_val = max(max_val, num)
            return min_val, max_val
        
        def calculate_max_diff(nums, x, y):
            prev = None
            max_diff = 0
            for num in nums:
                if num == -1:
                    if prev is not None:
                        max_diff = max(max_diff, abs(prev - x), abs(prev - y))
                    prev = min(x, y)
                else:
                    if prev is not None:
                        max_diff = max(max_diff, abs(prev - num))
                    prev = num
            return max_diff
        
        min_val, max_val = find_min_max(nums)
        
        if min_val == float('inf'):
            return 0
        
        x = min_val
        y = max_val
        
        return calculate_max_diff(nums, x, y)