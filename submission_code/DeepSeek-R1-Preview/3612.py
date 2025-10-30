from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        valid_starts = set()
        # Precompute all valid starting indices
        for a in range(n - k + 1):
            is_increasing = True
            for i in range(a, a + k - 1):
                if nums[i] >= nums[i + 1]:
                    is_increasing = False
                    break
            if is_increasing:
                valid_starts.add(a)
        # Check for two adjacent valid subarrays
        max_a = n - 2 * k
        for a in valid_starts:
            if a > max_a:
                continue
            if (a + k) in valid_starts:
                return True
        return False