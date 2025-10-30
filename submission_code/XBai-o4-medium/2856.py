from collections import defaultdict
from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_unique = len(set(nums))
        n = len(nums)
        count = 0
        
        for i in range(n):
            freq = defaultdict(int)
            current_unique = 0
            for j in range(i, n):
                num = nums[j]
                if freq[num] == 0:
                    current_unique += 1
                freq[num] += 1
                if current_unique == total_unique:
                    count += 1
        
        return count