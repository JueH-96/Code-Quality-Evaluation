from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_count = defaultdict(int)
        prefix_count[0] = 1
        current_count = 0
        interesting_subarrays = 0
        
        for num in nums:
            if num % modulo == k:
                current_count += 1
            
            target_count = (current_count - k) % modulo
            interesting_subarrays += prefix_count[target_count]
            prefix_count[current_count % modulo] += 1
        
        return interesting_subarrays