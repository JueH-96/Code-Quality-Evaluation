from collections import defaultdict
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # Split the array into runs of elements where (x & k) == k
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
        # Process each run
        for run in runs:
            prev = None
            run_total = 0
            for num in run:
                curr = defaultdict(int)
                # Count subarrays ending at current element with AND equal to num
                curr[num] += 1
                # Merge with previous AND values
                if prev is not None:
                    for and_val, count in prev.items():
                        new_and = and_val & num
                        curr[new_and] += count
                # Add the count of k in current AND values
                run_total += curr.get(k, 0)
                # Update prev to curr for the next iteration
                prev = curr.copy()
            total += run_total
        return total