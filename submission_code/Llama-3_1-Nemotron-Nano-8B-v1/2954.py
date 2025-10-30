from collections import defaultdict
from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k or k == 0:
            return 0
        
        freq = defaultdict(int)
        current_sum = 0
        for i in range(k):
            current_sum += nums[i]
            freq[nums[i]] += 1
        
        max_sum = current_sum if len(freq) >= m else 0
        
        for i in range(k, n):
            left = nums[i - k]
            freq[left] -= 1
            if freq[left] == 0:
                del freq[left]
            
            current_sum += nums[i] - left
            freq[nums[i]] += 1
            
            if len(freq) >= m:
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum