from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # This will store the count of prefix sums with a specific remainder
        prefix_count = defaultdict(int)
        # Initialize with 0 remainder having 1 count
        prefix_count[0] = 1
        
        # This will store the current prefix sum remainder
        current_remainder = 0
        # This will store the count of interesting subarrays
        interesting_count = 0
        
        for num in nums:
            # Check if the current number satisfies the condition
            if num % modulo == k:
                current_remainder += 1
            
            # We only care about the remainder with respect to modulo
            current_remainder %= modulo
            
            # Calculate the needed remainder to form an interesting subarray
            needed_remainder = (current_remainder - k + modulo) % modulo
            
            # Add the count of subarrays that end here and are interesting
            interesting_count += prefix_count[needed_remainder]
            
            # Update the prefix count with the current remainder
            prefix_count[current_remainder] += 1
        
        return interesting_count