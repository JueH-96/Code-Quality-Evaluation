from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return -1  # Cannot split a single-element array

        # Step 1: Find the dominant element
        freq = Counter(nums)
        dominant = None
        total_freq = 0
        for num, count in freq.items():
            if 2 * count > n:
                dominant = num
                total_freq = count
                break

        # Step 2: Find the smallest index i for a valid split
        count_left = 0
        for i in range(n - 1):
            if nums[i] == dominant:
                count_left += 1
            # Check dominance in left and right subarrays
            if 2 * count_left > i + 1:
                freq_right = total_freq - count_left
                len_right = n - (i + 1)
                if 2 * freq_right > len_right:
                    return i
        return -1