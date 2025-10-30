from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                left = nums[:i]
                right = nums[j+1:]
                combined = left + right
                valid = True
                for k in range(len(combined) - 1):
                    if combined[k] >= combined[k+1]:
                        valid = False
                        break
                if valid:
                    count += 1
        return count