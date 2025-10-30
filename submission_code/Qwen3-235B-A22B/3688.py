from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        # Compute the original Kadane's sum
        original_max = current = nums[0]
        for num in nums[1:]:
            current = max(num, current + num)
            original_max = max(original_max, current)
        
        unique_x = set(nums)
        best_with_deletion = -float('inf')
        
        for x in unique_x:
            current_sum = 0
            max_sub = -float('inf')
            has_element = False
            
            for num in nums:
                if num == x:
                    continue
                has_element = True
                current_sum = max(num, current_sum + num)
                max_sub = max(max_sub, current_sum)
            
            if has_element:
                best_with_deletion = max(best_with_deletion, max_sub)
        
        return max(original_max, best_with_deletion)