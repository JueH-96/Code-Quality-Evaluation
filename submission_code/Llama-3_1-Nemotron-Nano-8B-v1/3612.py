from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        valid_starts = []
        n = len(nums)
        # Generate all valid starting indices for strictly increasing subarrays of length k
        for i in range(n - k + 1):
            is_increasing = True
            for j in range(i, i + k - 1):
                if nums[j] >= nums[j + 1]:
                    is_increasing = False
                    break
            if is_increasing:
                valid_starts.append(i)
        # Check if any two consecutive valid starts are exactly k apart
        valid_set = set(valid_starts)
        for a in valid_starts:
            if (a + k) in valid_set:
                return True
        return False