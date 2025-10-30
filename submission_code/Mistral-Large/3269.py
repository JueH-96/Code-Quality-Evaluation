from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        m = len(pattern)
        n = len(nums)
        count = 0

        for i in range(n - m):
            match = True
            for k in range(m):
                if pattern[k] == 1 and nums[i + k + 1] <= nums[i + k]:
                    match = False
                    break
                if pattern[k] == 0 and nums[i + k + 1] != nums[i + k]:
                    match = False
                    break
                if pattern[k] == -1 and nums[i + k + 1] >= nums[i + k]:
                    match = False
                    break
            if match:
                count += 1

        return count