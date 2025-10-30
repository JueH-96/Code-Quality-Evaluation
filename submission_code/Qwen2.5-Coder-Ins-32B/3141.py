from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        n = len(nums)
        full_repeats = target // total_sum
        target %= total_sum
        
        min_length = float('inf')
        current_sum = 0
        left = 0
        
        for right in range(2 * n):
            current_sum += nums[right % n]
            
            while current_sum > target:
                current_sum -= nums[left % n]
                left += 1
            
            if current_sum == target:
                min_length = min(min_length, right - left + 1)
        
        return min_length + full_repeats * n if min_length != float('inf') else -1