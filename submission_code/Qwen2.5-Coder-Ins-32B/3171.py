from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        # If both arrays have no zeros and sums are not equal, return -1
        if zeros1 == 0 and zeros2 == 0 and sum1 != sum2:
            return -1
        
        # Calculate the minimum possible sum if we replace zeros with 1
        min_sum1 = sum1 + zeros1
        min_sum2 = sum2 + zeros2
        
        # If the minimum possible sum of one array is less than the other and it has no zeros, return -1
        if min_sum1 < min_sum2 and zeros1 == 0:
            return -1
        if min_sum2 < min_sum1 and zeros2 == 0:
            return -1
        
        # Return the maximum of the two minimum possible sums
        return max(min_sum1, min_sum2)