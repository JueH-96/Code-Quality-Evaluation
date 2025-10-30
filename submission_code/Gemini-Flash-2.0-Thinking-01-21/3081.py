from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        left_ptr = 0
        right_ptr = n // 2
        pairs_count = 0
        while left_ptr < n // 2 and right_ptr < n:
            if nums[left_ptr] < nums[right_ptr]:
                pairs_count += 1
                left_ptr += 1
                right_ptr += 1
            else:
                right_ptr += 1
        return n - 2 * pairs_count