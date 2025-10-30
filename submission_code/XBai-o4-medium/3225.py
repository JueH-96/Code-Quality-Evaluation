from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        counts = defaultdict(int)
        left = 0
        max_len = 0
        for right in range(len(nums)):
            num = nums[right]
            counts[num] += 1
            # Ensure the frequency of the current element does not exceed k
            while counts[num] > k:
                counts[nums[left]] -= 1
                left += 1
            # Update the maximum length of the valid subarray
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len