from collections import defaultdict
from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        d = None
        for num, count in freq.items():
            if 2 * count > n:
                d = num
                break
        
        total_d = freq[d]
        
        left_count = 0
        for i in range(n - 1):
            if nums[i] == d:
                left_count += 1
            
            left_len = i + 1
            right_len = n - left_len
            
            if left_count * 2 > left_len and (total_d - left_count) * 2 > right_len:
                return i
        
        return -1