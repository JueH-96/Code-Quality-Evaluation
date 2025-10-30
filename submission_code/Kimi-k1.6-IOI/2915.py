from collections import defaultdict
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # Initialize with prefix sum 0 having count 1
        current_sum = 0
        result = 0
        
        for num in nums:
            # Update current_sum based on whether the current number meets the condition
            if num % modulo == k:
                current_sum += 1
            
            # Calculate the target prefix sum modulo value needed to form a valid subarray ending at current index
            target = (current_sum - k) % modulo
            result += prefix_counts[target]
            
            # Update the prefix_counts with the current sum modulo modulo
            current_mod = current_sum % modulo
            prefix_counts[current_mod] += 1
        
        return result