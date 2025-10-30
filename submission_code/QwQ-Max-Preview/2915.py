from collections import defaultdict
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # Initial state with prefix sum 0
        current_sum = 0
        answer = 0
        
        for num in nums:
            # Update the current prefix sum if the current element meets the condition
            if num % modulo == k:
                current_sum += 1
            # Calculate the required residue for a valid previous prefix sum
            required_r = (current_sum - k) % modulo
            # Add the count of such residues from the hash map to the answer
            answer += prefix_counts.get(required_r, 0)
            # Update the hash map with the current residue of the prefix sum
            residue = current_sum % modulo
            prefix_counts[residue] += 1
        
        return answer