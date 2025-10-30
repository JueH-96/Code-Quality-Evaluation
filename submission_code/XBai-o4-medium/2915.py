from collections import defaultdict
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        remainder_counts = defaultdict(int)
        remainder_counts[0] = 1
        current_prefix = 0
        result = 0
        
        for num in nums:
            # Determine if the current number meets the condition
            a = 1 if (num % modulo == k) else 0
            # Update the current prefix sum
            current_prefix += a
            # Calculate the current remainder of the prefix sum modulo 'modulo'
            current_remainder = current_prefix % modulo
            # Calculate the target remainder we need to look for
            target_remainder = (current_remainder - k) % modulo
            # Add the count of the target remainder to the result
            result += remainder_counts.get(target_remainder, 0)
            # Update the count of the current remainder in the dictionary
            remainder_counts[current_remainder] += 1
        
        return result