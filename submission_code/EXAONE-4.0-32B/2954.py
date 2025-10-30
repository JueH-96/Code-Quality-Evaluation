from collections import defaultdict
from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        current_sum = 0
        freq = defaultdict(int)
        
        for i in range(k):
            current_sum += nums[i]
            freq[nums[i]] += 1
        
        max_sum = 0
        if len(freq) >= m:
            max_sum = current_sum
        
        for i in range(k, n):
            left_num = nums[i - k]
            current_sum -= left_num
            freq[left_num] -= 1
            if freq[left_num] == 0:
                del freq[left_num]
            
            right_num = nums[i]
            current_sum += right_num
            freq[right_num] += 1
            
            if len(freq) >= m:
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum