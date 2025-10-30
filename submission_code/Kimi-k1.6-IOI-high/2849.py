from bisect import bisect_left, insort
from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        for R in range(n):
            current_sorted = []
            current_imbalance = 0
            for L in range(R, -1, -1):
                num = nums[L]
                idx = bisect_left(current_sorted, num)
                
                # Calculate old_gap before insertion
                old_gap = 0
                if idx > 0 and idx < len(current_sorted):
                    left_val = current_sorted[idx-1]
                    right_val = current_sorted[idx]
                    if right_val - left_val > 1:
                        old_gap = 1
                
                # Insert num into current_sorted
                insort(current_sorted, num)
                
                # Calculate new gaps after insertion
                new_gap1 = 0
                if idx > 0:
                    left_val = current_sorted[idx-1]
                    if num - left_val > 1:
                        new_gap1 = 1
                
                new_gap2 = 0
                if idx < len(current_sorted) - 1:
                    right_val = current_sorted[idx + 1]
                    if right_val - num > 1:
                        new_gap2 = 1
                
                # Update current_imbalance
                delta = new_gap1 + new_gap2 - old_gap
                current_imbalance += delta
                total += current_imbalance
        
        return total