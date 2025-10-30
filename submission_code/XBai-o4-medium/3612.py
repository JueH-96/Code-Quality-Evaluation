from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        max_a = n - 2 * k
        for a in range(max_a + 1):
            # Check first subarray starting at a
            valid_first = True
            for i in range(a, a + k - 1):
                if nums[i] >= nums[i + 1]:
                    valid_first = False
                    break
            if not valid_first:
                continue
            # Check second subarray starting at a + k
            start_second = a + k
            valid_second = True
            for i in range(start_second, start_second + k - 1):
                if nums[i] >= nums[i + 1]:
                    valid_second = False
                    break
            if valid_second:
                return True
        return False