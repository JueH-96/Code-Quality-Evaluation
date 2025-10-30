from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        count = 0
        n = len(nums)
        m = len(pattern)
        # The number of possible starting indices is n - m
        for i in range(n - m):
            valid = True
            for k in range(m):
                a = nums[i + k]
                b = nums[i + k + 1]
                p = pattern[k]
                if p == 1:
                    if not (b > a):
                        valid = False
                        break
                elif p == 0:
                    if not (b == a):
                        valid = False
                        break
                elif p == -1:
                    if not (b < a):
                        valid = False
                        break
            if valid:
                count += 1
        return count