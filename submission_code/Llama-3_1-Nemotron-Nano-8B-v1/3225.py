from collections import defaultdict
from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = defaultdict(int)
        max_length = 0
        left = 0
        over = set()
        
        for right in range(len(nums)):
            num = nums[right]
            freq[num] += 1
            if freq[num] == k + 1:
                over.add(num)
            
            # Shrink the window until all elements in 'over' are removed
            while over:
                left_num = nums[left]
                if freq[left_num] == k + 1:
                    over.remove(left_num)
                freq[left_num] -= 1
                left += 1
            
            # Update the maximum length
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        
        return max_length