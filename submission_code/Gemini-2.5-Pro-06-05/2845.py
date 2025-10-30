from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        """
        Calculates the minimum value of a partition for the given array.

        The value of a partition (nums1, nums2) is |max(nums1) - min(nums2)|.
        The key insight is that for any partition, if a = max(nums1) and b = min(nums2),
        no element of nums can exist strictly between a and b. This implies that a and b
        must be adjacent elements in the sorted version of nums.
        
        Therefore, the problem reduces to finding the minimum difference between any
        two adjacent elements in the sorted array.
        """

        # Sort the array to bring elements with potentially small differences together.
        nums.sort()
        
        # The minimum partition value will be the minimum difference between any
        # two adjacent elements. We can find this with a single pass.
        # A generator expression is a concise and Pythonic way to do this.
        return min(nums[i] - nums[i-1] for i in range(1, len(nums)))