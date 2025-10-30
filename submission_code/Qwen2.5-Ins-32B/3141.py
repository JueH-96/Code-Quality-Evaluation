from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        n = len(nums)
        nums = nums * 3  # Extend nums to cover at least one full cycle plus some extra
        min_len = float('inf')
        
        left = 0
        current_sum = 0
        
        for right in range(len(nums)):
            current_sum += nums[right]
            
            while current_sum >= target:
                if current_sum == target:
                    min_len = min(min_len, right - left + 1)
                current_sum -= nums[left]
                left += 1
        
        if min_len == float('inf'):
            return -1
        
        # Adjust for the case where the subarray wraps around the end of the original array
        full_cycles = target // total_sum
        remaining_target = target % total_sum
        if remaining_target == 0:
            return full_cycles * n
        
        return full_cycles * n + min_len