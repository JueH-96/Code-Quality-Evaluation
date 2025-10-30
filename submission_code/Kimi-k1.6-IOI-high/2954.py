from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        freq = defaultdict(int)
        max_sum = 0
        current_sum = 0
        left = 0
        
        for right in range(len(nums)):
            num = nums[right]
            current_sum += num
            freq[num] += 1
            
            # Maintain the window size of k
            while (right - left + 1) > k:
                left_num = nums[left]
                current_sum -= left_num
                freq[left_num] -= 1
                if freq[left_num] == 0:
                    del freq[left_num]
                left += 1
            
            # Check if current window is valid and update max_sum
            if (right - left + 1) == k and len(freq) >= m:
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum