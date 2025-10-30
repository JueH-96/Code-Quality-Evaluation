from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        # If there are no zeros and the sums are already equal, return the sum
        if zeros1 == 0 and zeros2 == 0 and sum1 == sum2:
            return sum1
        
        # If one array has no zeros and its sum is less than the other array's sum plus its zeros
        if (zeros1 == 0 and sum1 < sum2 + zeros2) or (zeros2 == 0 and sum2 < sum1 + zeros1):
            return -1
        
        # Calculate the target sum
        target_sum = max(sum1 + zeros1, sum2 + zeros2)
        
        return target_sum