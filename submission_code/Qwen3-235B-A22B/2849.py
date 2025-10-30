from bisect import bisect_left, insort_left
from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        for j in range(n):
            sorted_list = []
            curr_imbalance = 0
            sum_total = 0
            for i in range(j, -1, -1):
                x = nums[i]
                index = bisect_left(sorted_list, x)
                # Check if x already exists in the sorted_list
                if index < len(sorted_list) and sorted_list[index] == x:
                    sum_total += curr_imbalance
                    continue
                # Determine predecessor and successor
                pre = None
                suc = None
                if index > 0:
                    pre = sorted_list[index - 1]
                if index < len(sorted_list):
                    suc = sorted_list[index]
                # Calculate old_gap and new_gaps
                old_gap = 0
                new_gaps = 0
                if pre is not None and suc is not None:
                    old_gap = 1 if (suc - pre) > 1 else 0
                    new_gaps += 1 if (x - pre) > 1 else 0
                    new_gaps += 1 if (suc - x) > 1 else 0
                elif pre is not None:
                    new_gaps += 1 if (x - pre) > 1 else 0
                elif suc is not None:
                    new_gaps += 1 if (suc - x) > 1 else 0
                # Update current imbalance
                delta = new_gaps - old_gap
                curr_imbalance += delta
                # Insert x into the sorted list
                insort_left(sorted_list, x)
                # Add current imbalance to sum_total
                sum_total += curr_imbalance
            total += sum_total
        return total