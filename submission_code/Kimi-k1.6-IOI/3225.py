from collections import defaultdict
from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = defaultdict(int)
        left = 0
        max_len = 0
        for right, num in enumerate(nums):
            freq[num] += 1
            # Ensure the frequency of the current num does not exceed k
            while freq[num] > k:
                left_num = nums[left]
                freq[left_num] -= 1
                left += 1
            # Update the maximum length found
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len