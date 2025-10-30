from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count = 0
        prefix_count = defaultdict(int)
        prefix_count[0] = 1
        current_count = 0
        
        for num in nums:
            if num % modulo == k:
                current_count += 1
            count += prefix_count[(current_count % modulo - k) % modulo]
            prefix_count[current_count % modulo] += 1
        
        return count