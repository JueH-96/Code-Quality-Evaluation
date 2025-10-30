from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        first_val = nums[0]
        all_equal = True
        for x in nums:
            if x != first_val:
                all_equal = False
                break
        if all_equal:
            return 0
            
        unique_values = set(nums)
        min_seconds = float('inf')
        
        for target_value in unique_values:
            current_nums = list(nums)
            seconds = 0
            while True:
                all_target = True
                for x in current_nums:
                    if x != target_value:
                        all_target = False
                        break
                if all_target:
                    min_seconds = min(min_seconds, seconds)
                    break
                if seconds >= n: # Optimization to avoid infinite loop in some cases, though should not be needed in theory for this problem
                    break
                seconds += 1
                next_nums = [0] * n
                for i in range(n):
                    options = [current_nums[i], current_nums[(i - 1 + n) % n], current_nums[(i + 1) % n]]
                    if target_value in options:
                        next_nums[i] = target_value
                    else:
                        next_nums[i] = current_nums[i]
                current_nums = next_nums
                
        return min_seconds