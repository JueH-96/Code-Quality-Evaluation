from collections import defaultdict
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        prev = defaultdict(int)
        for num in nums:
            current = defaultdict(int)
            # Add the subarray consisting of the current element only
            current[num] += 1
            # Process all previous AND values
            for val, cnt in prev.items():
                new_val = val & num
                current[new_val] += cnt
            # Add the count of current subarrays with AND equal to k
            total += current[k]
            # Update prev for the next iteration
            prev = current
        return total