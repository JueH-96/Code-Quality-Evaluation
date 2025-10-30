from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        m = len(nums) // 2
        median = nums[m]
        
        if median == k:
            return 0
        elif median < k:
            total = 0
            for i in range(m, len(nums)):
                if nums[i] < k:
                    total += (k - nums[i])
                else:
                    break  # Since the array is sorted, no more elements <k
            return total
        else:
            total = 0
            for i in range(m + 1):
                if nums[i] > k:
                    total += (nums[i] - k)
            return total