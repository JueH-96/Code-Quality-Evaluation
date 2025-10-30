from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        freq = defaultdict(int)
        freq[0] = 1  # Initialize with prefix sum 0
        current_sum = 0
        count = 0
        
        for num in nums:
            if num % modulo == k:
                current_sum += 1
            target = (current_sum - k) % modulo
            count += freq[target]
            current_sum_mod = current_sum % modulo
            freq[current_sum_mod] += 1
        
        return count