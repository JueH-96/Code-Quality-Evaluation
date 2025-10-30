from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)

        def is_increasing(arr: List[int]) -> bool:
            for i in range(1, len(arr)):
                if arr[i] <= arr[i - 1]:
                    return False
            return True

        for a in range(n - 2 * k + 1):
            b = a + k
            subarray1 = nums[a:a + k]
            subarray2 = nums[b:b + k]
            if is_increasing(subarray1) and is_increasing(subarray2):
                return True

        return False