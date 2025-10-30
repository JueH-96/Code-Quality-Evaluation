from typing import List
from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # Split the array into runs where each element x satisfies (x & k) == k
        runs = []
        current_run = []
        for num in nums:
            if (num & k) == k:
                current_run.append(num)
            else:
                if current_run:
                    runs.append(current_run)
                    current_run = []
        if current_run:
            runs.append(current_run)
        
        total = 0
        for run in runs:
            current_ands = {}
            for num in run:
                new_ands = defaultdict(int)
                # Add the subarray consisting of just the current element
                new_ands[num] += 1
                # Extend previous subarrays with the current element
                for and_val, count in current_ands.items():
                    new_and = and_val & num
                    new_ands[new_and] += count
                # Update current_ands for the next iteration
                current_ands = new_ands
                # Add the count of subarrays with AND equal to k
                if k in current_ands:
                    total += current_ands[k]
        return total