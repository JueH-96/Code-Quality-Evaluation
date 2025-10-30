from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        # Count the maximum frequency of any value in the sorted array
        maxfreq = 1
        cur = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                cur += 1
            else:
                if cur > maxfreq:
                    maxfreq = cur
                cur = 1
        if cur > maxfreq:
            maxfreq = cur

        # If the largest group is more than half of the array, some of its items
        # cannot be paired. Otherwise, we can pair everything except at most one
        # element when n is odd.
        return max(2 * maxfreq - n, n & 1)