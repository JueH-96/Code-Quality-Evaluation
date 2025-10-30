from typing import List
import bisect

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        for i in range(n):
            sorted_unique = []
            imbalance = 0
            for j in range(i, n):
                num = nums[j]
                pos = bisect.bisect_left(sorted_unique, num)
                # Check if num is already present
                if pos < len(sorted_unique) and sorted_unique[pos] == num:
                    # Duplicate, no change
                    total_sum += imbalance
                    continue
                # Calculate changes to imbalance
                left_gap = right_gap = 0
                if pos > 0:
                    if num - sorted_unique[pos - 1] > 1:
                        left_gap = 1
                if pos < len(sorted_unique):
                    if sorted_unique[pos] - num > 1:
                        right_gap = 1
                # If there are both left and right elements and the gap was previously >1, decrement imbalance
                if pos > 0 and pos < len(sorted_unique):
                    if sorted_unique[pos] - sorted_unique[pos - 1] > 1:
                        imbalance -= 1
                # Update imbalance with new gaps
                imbalance += left_gap + right_gap
                # Insert num into sorted_unique
                bisect.insort(sorted_unique, num)
                # Add current imbalance to total_sum
                total_sum += imbalance
        return total_sum