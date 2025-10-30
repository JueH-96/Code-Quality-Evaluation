from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_count = defaultdict(int)
        prefix_count[0] = 1
        count = 0
        current_sum = 0
        
        for num in nums:
            if num % modulo == k:
                current_sum += 1
            current_sum %= modulo
            count += prefix_count[(current_sum - k) % modulo]
            prefix_count[current_sum] += 1
        
        return count