from collections import defaultdict
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # Initial state with prefix sum 0
        current_sum = 0
        result = 0
        
        for num in nums:
            # Determine if the current element contributes to the count
            hit = 1 if (num % modulo) == k else 0
            current_sum += hit
            current_mod = current_sum % modulo
            
            # Calculate the required modulo value from previous prefix sums
            required_mod = (current_mod - k) % modulo
            
            # Add the count of prefix sums that match the required modulo
            result += prefix_counts[required_mod]
            
            # Update the prefix counts with the current modulo value
            prefix_counts[current_mod] += 1
        
        return result