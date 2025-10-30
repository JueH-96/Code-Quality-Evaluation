from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        original_max = self.kadane(nums)
        unique_x = set(nums)
        max_sum = original_max
        
        for x in unique_x:
            modified = [num for num in nums if num != x]
            if not modified:
                continue
            current_max = self.kadane(modified)
            if current_max > max_sum:
                max_sum = current_max
        
        return max_sum
    
    def kadane(self, arr):
        max_current = max_global = arr[0]
        for num in arr[1:]:
            max_current = max(num, max_current + num)
            if max_current > max_global:
                max_global = max_current
        return max_global