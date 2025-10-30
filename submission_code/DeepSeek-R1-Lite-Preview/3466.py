from typing import List
from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        prev_freq = defaultdict(int)
        total = 0
        for num in nums:
            current_freq = defaultdict(int)
            # AND the new number with all previous AND values
            for and_val, count in prev_freq.items():
                new_and = and_val & num
                current_freq[new_and] += count
            # Add the new number as a subarray of length 1
            current_freq[num] += 1
            # Add the count of k in the current frequency to the total
            total += current_freq.get(k, 0)
            # Set prev_freq to current_freq for the next iteration
            prev_freq = current_freq
        return total