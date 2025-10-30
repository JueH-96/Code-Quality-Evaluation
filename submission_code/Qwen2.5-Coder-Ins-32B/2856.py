from typing import List
from collections import Counter

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_distinct = len(set(nums))
        count = 0
        n = len(nums)
        
        for start in range(n):
            current_counter = Counter()
            for end in range(start, n):
                current_counter[nums[end]] += 1
                if len(current_counter) == total_distinct:
                    count += 1
        
        return count