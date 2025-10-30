from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        count = [0] * (n + 1)
        last = defaultdict(int)
        distinct_count = 0
        total_subarrays = n * (n + 1) // 2
        
        for i in range(n):
            last_occurrence = last[nums[i]]
            distinct_count += i - last_occurrence + 1
            count[distinct_count] += 1
            last[nums[i]] = i + 1
        
        median_position = (total_subarrays + 1) // 2
        current_count = 0
        
        for i in range(1, n + 1):
            current_count += count[i]
            if current_count >= median_position:
                return i