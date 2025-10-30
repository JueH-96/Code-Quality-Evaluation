from typing import List
from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        prev_counts = defaultdict(int)
        for num in nums:
            current_counts = defaultdict(int)
            # The subarray consisting of the current number alone
            current_counts[num] += 1
            # Extend previous subarrays with the current number
            for and_val, cnt in prev_counts.items():
                new_and = and_val & num
                current_counts[new_and] += cnt
            # Add the count of subarrays ending here with AND equal to k
            total += current_counts.get(k, 0)
            # Update prev_counts for the next iteration
            prev_counts = current_counts
        return total