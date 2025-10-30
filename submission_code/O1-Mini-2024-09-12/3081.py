from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        right = n // 2
        max_k = 0
        while left <= right:
            mid = (left + right) // 2
            possible = True
            for i in range(mid):
                if nums[i] >= nums[n - mid + i]:
                    possible = False
                    break
            if possible:
                max_k = mid
                left = mid + 1
            else:
                right = mid - 1
        return n - 2 * max_k