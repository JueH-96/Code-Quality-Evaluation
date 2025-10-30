from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev_ors = []
        for num in nums:
            current_ors = [num]
            for prev in prev_ors:
                current_ors.append(prev | num)
            # Remove duplicates
            current_ors = list(set(current_ors))
            # Update the minimum difference
            for val in current_ors:
                diff = abs(val - k)
                if diff < min_diff:
                    min_diff = diff
            # Update prev_ors for next iteration
            prev_ors = current_ors
        return min_diff