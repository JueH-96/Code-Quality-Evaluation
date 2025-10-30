from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev_or = []  # List to hold OR values of subarrays ending at previous element
        for num in nums:
            current_or = {num}  # Start with the subarray containing only the current element
            for val in prev_or:
                current_or.add(val | num)  # Combine previous OR values with current element
            # Update the minimum difference
            for val in current_or:
                diff = abs(val - k)
                if diff < min_diff:
                    min_diff = diff
            # Prepare for next iteration
            prev_or = list(current_or)
        return min_diff