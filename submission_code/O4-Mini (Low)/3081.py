from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        """
        We want to maximize the number of pairs (i, j) with i<j and nums[i] < nums[j].
        Since nums is sorted non-decreasingly, we can greedily match the smallest
        elements with the smallest possible larger elements from the right half.
        """
        n = len(nums)
        # Two-pointer approach:
        # l scans the left part [0 .. n//2 - 1], r scans from n//2 .. n-1
        l = 0
        r = n // 2
        matches = 0
        
        # Try to greedily match nums[l] with the earliest nums[r] > nums[l]
        while l < n // 2 and r < n:
            if nums[l] < nums[r]:
                matches += 1
                l += 1
                r += 1
            else:
                # nums[r] is not strictly greater, move r forward
                r += 1
        
        # Each match removes 2 elements
        return n - 2 * matches