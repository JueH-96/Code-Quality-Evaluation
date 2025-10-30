from collections import defaultdict
from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        count = defaultdict(int)
        max_length = 0
        left = 0
        for right in range(len(nums)):
            num = nums[right]
            count[num] += 1
            # Check if the current element's count exceeds k
            while count[num] > k:
                # Move the left pointer to reduce the count
                left_num = nums[left]
                count[left_num] -= 1
                left += 1
            # Update the maximum length of the valid subarray
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        return max_length