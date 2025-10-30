from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # Helper function to calculate the maximum difference
        # after replacing -1 with x or y
        def calculate_max_diff(nums, x, y):
            prev = None
            max_diff = 0
            for num in nums:
                if num == -1:
                    if prev is not None:
                        max_diff = max(max_diff, abs(prev - x), abs(prev - y))
                    prev = x  # or y, doesn't matter as we are checking both
                else:
                    if prev is not None:
                        max_diff = max(max_diff, abs(prev - num))
                    prev = num
            return max_diff

        # Collect all non-negative numbers
        non_negatives = [num for num in nums if num != -1]

        # If there are no non-negative numbers, we can make all -1 the same number
        if not non_negatives:
            return 0

        # Find the minimum and maximum of the non-negative numbers
        min_val = min(non_negatives)
        max_val = max(non_negatives)

        # The best choice for x and y is around the median of the range
        # between min_val and max_val to minimize the maximum difference
        # We try to place x and y around the middle of the range
        mid_val = (min_val + max_val) // 2

        # Try different combinations of x and y around the mid_val
        candidates = [
            (mid_val, mid_val),
            (mid_val, mid_val + 1),
            (mid_val - 1, mid_val),
            (mid_val - 1, mid_val + 1),
            (mid_val + 1, mid_val + 1)
        ]

        # Calculate the minimum possible maximum difference
        min_max_diff = float('inf')
        for x, y in candidates:
            min_max_diff = min(min_max_diff, calculate_max_diff(nums, x, y))

        return min_max_diff