from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # Use a difference array to count how many queries cover each index
        diff = [0] * (n + 1)
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1

        # Build the coverage count and check against nums[i]
        curr = 0
        for i in range(n):
            curr += diff[i]
            # If the total number of covering queries is less than the needed decrements, fail
            if curr < nums[i]:
                return False

        return True