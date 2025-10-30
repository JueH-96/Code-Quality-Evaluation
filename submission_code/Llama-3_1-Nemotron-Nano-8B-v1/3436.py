from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev_ors = []
        for num in nums:
            current_ors = {num}
            for or_val in prev_ors:
                current_ors.add(or_val | num)
            for or_val in current_ors:
                diff = abs(or_val - k)
                if diff < min_diff:
                    min_diff = diff
            prev_ors = list(current_ors)
        return min_diff