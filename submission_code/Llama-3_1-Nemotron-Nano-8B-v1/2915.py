from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count_map = defaultdict(int)
        count_map[0] = 1  # Initialize with prefix sum 0
        current_sum = 0
        result = 0
        
        for num in nums:
            # Check if the current number satisfies the condition
            if num % modulo == k:
                current_sum += 1
            
            # Calculate the current remainder and target remainder
            remainder = current_sum % modulo
            target = (remainder - k) % modulo
            
            # Add the count of target remainder seen so far
            result += count_map.get(target, 0)
            
            # Update the count_map with the current remainder
            count_map[remainder] += 1
        
        return result