from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count_dict = defaultdict(int)
        count_dict[0] = 1  # initial prefix sum is 0
        current_sum = 0
        result = 0
        
        for num in nums:
            if num % modulo == k:
                current_sum += 1
            # Calculate the required residue (current_sum - k) mod modulo
            required = (current_sum - k) % modulo
            result += count_dict[required]
            
            # Update the current residue (current_sum mod modulo) in the dictionary
            current_residue = current_sum % modulo
            count_dict[current_residue] += 1
        
        return result