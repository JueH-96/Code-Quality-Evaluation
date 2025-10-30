from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        """
        The minimum possible value of |max(nums1) - min(nums2)| is obtained by
        sorting the array and taking the smallest difference between two
        consecutive elements.  Placing every element up to the first of those
        two into nums1 and the rest into nums2 satisfies the required
        conditions and achieves that difference.
        """
        nums.sort()
        best = float('inf')
        for i in range(1, len(nums)):
            best = min(best, nums[i] - nums[i - 1])
        return best