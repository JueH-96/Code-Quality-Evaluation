from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        n = len(nums)
        total = n * (n + 1) // 2
        
        bad = 0
        left = 0
        current_count = 0
        
        for right in range(n):
            if nums[right] == max_num:
                current_count += 1
            
            # Move left to ensure current_count < k
            while current_count >= k:
                if nums[left] == max_num:
                    current_count -= 1
                left += 1
            
            # Add the number of valid subarrays ending at 'right'
            bad += (right - left + 1)
        
        return total - bad